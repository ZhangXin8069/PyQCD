# Gluon 裸矩阵元：ratio 与动量方向 average 的运算顺序测试

记录日期：2026-11-07。本文按实际代码、保存的 bootstrap／拟合产物及完成回执整理。
本次记录没有重新计算 OPE、C3 或拟合；原始单方向结果和三方向测试产物均保留。

## 1. 测试目的与范围

本测试回答：对相同动量大小的不同空间方向，先平均相关函数再构造 C3/C2，
与先逐方向构造 C3/C2 再平均，是否会改变提取的裸矩阵元及统计误差？

分析对象是 F32P30 的 HYP T-only gluon unpolarized 与 helicity disconnected 信号。
这里的 ratio 指 ordinary C3/C2，不是 self-ratio；两种方向平均顺序的比较，
也不是先分别拟合各方向矩阵元再平均。当前结果仍是未重整化的原生格点裸幅度，
不是已经完成重整化、算符混合扣除、LaMET 匹配和连续极限的物理 gluon PDF。

本轮只用三个正动量方向：

\[
\boldsymbol k_x=(3,0,0),\qquad
\boldsymbol k_y=(0,3,0),\qquad
\boldsymbol k_z=(0,0,3),\qquad w_d=\frac13.
\]

没有加入负动量方向，不把正反 link 的奇偶组合称为六动量方向平均。
三个方向采用固定的正循环旋转约定，不根据信号好坏翻转某个方向的符号。

## 2. 起点、数据与固定条件

### 2.1 原单方向测试与新测试目录

原先的 ratio／self-ratio 测试只使用正 z 动量和 `zdir`：

`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/SelfRatio_F32P30_20261106`。

因此原先的图没有进行动量方向平均，不能从该图推断两种平均顺序的差异。
本次另建独立目录：

`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/F32P30_DirectionOrder_20261107`。

下文用 `BASE` 指代这个新目录。原分析、原图及原 bootstrap 索引均未改写。

### 2.2 输入与共同配置

- OPE 根目录：`/public/group/lqcd/donghx/Ope_Gluon/Result_hyp_4D_10times/F32P30`，读取 `xdir`、`ydir`、`zdir`。
- 2pt 根目录：`/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96`，读取 `momsmear2x`、`momsmear2y`、`momsmear2z`。
- 输入清单及每个文件的 SHA256：`BASE/results/input_manifest.json`。
- 配置清单：`BASE/results/configurations.txt`。

以原正 z 方向 2pt 目录中的 801 个候选配置为起点，要求三个方向、两个通道所需
OPE 与 2pt 同时完整，最终共同配置为 777 个；24 个未满足输入条件的配置不使用，
具体原因逐配置保存在输入清单中。共同清单与原单方向分析完全一致。

| 项目 | 固定设置 |
|---|---|
| Ensemble | F32P30，\(32^3\times96\) |
| 格距 | \(a=0.0775\,\mathrm{fm}\) |
| 动量 | 三个正轴，模编号为 3，\(a|\boldsymbol P|=2\pi\cdot3/32\) |
| 位移 | \(n=z/a=0,\ldots,23\)；各轴均以沿 link 的正位移编号表示 |
| 配置／bootstrap | \(N_{\mathrm{cfg}}=777\)，\(N_b=3000\)，seed 为 1115 |
| 每个样本的抽样数 | 从共同清单有放回抽取 777 个配置 |
| 分箱 | 未分箱，尚未控制 Markov 链自相关误差 |
| 算符 | 两通道都只用 T，OPE 在关联前取实 |
| 源汇距 | \(T/a=10,\ldots,16\) |
| 插入时间 cut | \(c=2,3,4\)，保留 \(c\le\tau/a\le T/a-c\) |
| 拟合模型 | 逐位移的单参数相关常数模型 |

最小源汇距严格满足此前要求：

\[
\frac{T_{\min}}a
=\left\lfloor\frac{0.7\,\mathrm{fm}}{0.0775\,\mathrm{fm}}\right\rfloor+1
=10,\qquad T_{\min}=0.775\,\mathrm{fm}>0.7\,\mathrm{fm}.
\]

## 3. 算符旋转、极化和 C3 构造

### 3.1 三方向的 T 组合与投影

统一 OPE 的指标约定为 \(0123=xyzt\)。表中的每个分量对 unpolarized 表示
同指标的两个场强相乘，对 helicity 表示第二个场强取对偶：

| link／动量方向 | T 的两个分量 | helicity 分子使用的 2pt |
|---|---|---|
| 正 x | 31、32 | `pol15` |
| 正 y | 32、30 | `pol25` |
| 正 z | 30、31 | `pol35` |

例如正 x 的 T 组合为

\[
T_{U,x}=O(F_{31}F_{31})+O(F_{32}F_{32}),\qquad
T_{H,x}=O(F_{31}\widetilde F_{31})+O(F_{32}\widetilde F_{32}).
\]

所有组合都保留实际 link／空间求和结构，此处只简写张量指标。
沿用原测试的整体符号和无半因子的奇偶约定：

\[
O_{U,d}(n)=-2\operatorname{Re}T_{U,d,+}(n),\qquad
O_{H,d}(n)=\operatorname{Re}\bigl[T_{H,d,+}(n)-T_{H,d,-}(n)\bigr].
\]

unpolarized 使用无半因子的偶组合；helicity 使用无半因子的奇组合，局域点精确置零。
这与动量方向平均是两层不同运算。

### 3.2 每个 bootstrap 内先逐方向完成真空扣除

令 \(s\) 为源时间，\(\langle\cdot\rangle_b\) 为第 b 个样本的配置平均，
\(B_{d,X,c}(T,s)\) 为相应方向与核子投影的两点函数。逐方向计算

\[
\mathcal C_{3,d,X}^{(b)}(n,T,\tau)
=\frac1{N_t}\sum_s\left[
\left\langle O_{X,d}(s+\tau,n)B_{d,X}(T,s)\right\rangle_b
-\left\langle O_{X,d}(s+\tau,n)\right\rangle_b
 \left\langle B_{d,X}(T,s)\right\rangle_b\right].
\]

U 分子配非极化 2pt，H 分子配表中的完整复数极化 2pt，随后投影为

\[
N_{d,U}^{(b)}=\operatorname{Re}\mathcal C_{3,d,U}^{(b)},\qquad
N_{d,H}^{(b)}=\operatorname{Im}\mathcal C_{3,d,H}^{(b)}.
\]

两通道的 ordinary ratio 分母都使用非极化 2pt 实部：

\[
D_d^{(b)}(T)=\operatorname{Re}\left[
\frac1{N_t}\sum_s\left\langle B_{d,\mathrm{nopol}}(T,s)\right\rangle_b\right].
\]

因此 helicity 确实用极化 2pt 构造分子并取虚部，但不以近零的极化 C2 均值归一化。
时间索引按现有分析模 \(N_t\) 处理，没有另加边界折叠符号。

## 4. 两种平均顺序的精确定义

下面对每个通道分别使用上述实分子 \(N_d^{(b)}\)，省略共同的位移和时间指标。

### 4.1 A：先平均相关函数，再做 ratio

\[
R_A^{(b)}
=\frac{\sum_dw_dN_d^{(b)}}{\sum_dw_dD_d^{(b)}}
=\frac{\sum_dN_d^{(b)}}{\sum_dD_d^{(b)}}.
\]

代码名称为 `average_then_ratio`。这里平均的是已经完成同方向收缩、真空扣除和
投影的 C3，不是分别平均 OPE 和 2pt 后再相乘。后者会引入未登记的交叉方向项，
不是本测试的估计量。

### 4.2 B：先逐方向做 ratio，再平均

\[
r_d^{(b)}=\frac{N_d^{(b)}}{D_d^{(b)}},\qquad
R_B^{(b)}=\sum_dw_dr_d^{(b)}=\frac13\sum_d\frac{N_d^{(b)}}{D_d^{(b)}}.
\]

代码名称为 `ratio_then_average`。两方案都在每个 bootstrap 内先形成最终观测量，
然后再拟合；不是逐配置先求比值后平均，也不是先独立拟合三个方向。

### 4.3 为什么两种顺序一般不相同

固定样本和时间点，令 \(\overline D=\sum_dw_dD_d\)、
\(\overline r=\sum_dw_dr_d\)，当分母有定义时有精确恒等式

\[
R_A-R_B
=\frac{\sum_dw_d(D_d-\overline D)(r_d-\overline r)}{\overline D}.
\]

所以 A 相当于按各方向 C2 改变有效方向权重，而 B 是等权 ratio 平均。
若各方向分母严格相同，两者在每个样本中严格相同；否则有限统计的方向涨落和
比值非线性可能造成差异。上式是三个方向上的代数关系，不是对配置 ensemble
统计协方差的额外假设。哪一种误差较小、模型依赖较弱，需要实测，不能预先假定。

## 5. Bootstrap、分母检查与共同拟合

### 5.1 配对重采样与有效时间网格

本轮逐项复用了原正 z 测试的 `bootstrap_indices.npz`：所有方向、通道、位移、
源汇距以及两个平均顺序共享相同的配置索引，而不是各方向独立抽样。
每个样本重新计算 C3 的配置平均和真空扣除项，也重新计算 C2，再进行上述非线性比值。
3000 是重采样次数，不是独立规范场数量，不会凭空增加原始统计信息。

分母检查沿用 SNR、样本同号比例、近零分位数和有限样本比例门槛。
最终拟合有效时间网格取三个方向 C2 与平均 C2 分母门的共同交集，不因某方向失败
而临时改为平均剩余方向。本轮三个方向 C2 在全部 7 个源汇距均通过检查。

非物理插入时间 padding 保留为 NaN，并保存 `valid_tau_mask` 与 `ratio_valid_mask`；
有限但未通过分母门的值只作诊断，不静默删掉过零样本或更换方向权重。

### 5.2 相同模型，但不强制相同协方差与 AIC 权重

逐个位移联合窗口中的不同源汇距和插入时间，使用

\[
R_A(n,T,\tau)=M_A(n),\qquad R_B(n,T,\tau)=M_B(n).
\]

每个候选都只有一个自由常数，没有显式激发态指数项，也不是 summation 拟合。
固定最小源汇距为 10，上端扫描 11 到 16，对称 cut 为 2、3、4。
两种顺序限制到共同可计算候选，但观测量不同，所以各自估计协方差、质量和权重。

协方差采用 Schaefer–Strimmer correlation-to-identity 收缩，SVD 阈值为
\(10^{-12}\)。质量门要求正自由度、有限性、可用 bootstrap 比例不低于
\(0.9\)，以及

\[
Q\ge0.05,\qquad \chi^2/\mathrm{dof}\le3.
\]

固定 cut 内进行登记的 omission-penalized AIC 加权：

\[
\mathrm{AIC}_{i,b}=\chi^2_{i,b}+2k+2N_{\mathrm{omit},i},\qquad k=1.
\]

中心值与每个 bootstrap 样本分别选择／加权候选，再按既定规则组合可用 cut。
没有通过质量门时，保留带标记的 fallback 供诊断，不把它当正式通过结果。
“共同候选”不表示两方案的最终 AIC 权重必须相同。

误差是样本标准差，不再除以重采样次数的平方根：

\[
\sigma(M)=\sqrt{\frac1{N_b-1}\sum_b(M^{(b)}-\overline M)^2}.
\]

## 6. 实施经过与作业记录

| 阶段 | 实际操作和状态 |
|---|---|
| 输入及约定核对 | 确认三方向 OPE 张量轮换和对应的 nopol／pol15／pol25／pol35 文件，不直接把单方向图称作方向平均 |
| 独立实现 | 新建测试目录及 `run.py`、`config.json`、`cpu_job.sh`、`README.md`，旧分析不覆盖 |
| 提交前校验 | 合成数据检验顺序不交换及相同分母时相等；相关常数拟合恢复；真实 conf1000 三方向输入 shape；旧 bootstrap 来源回执 |
| 首次作业 | `3945195` 于 2026-11-07 22:15:04 提交，在输入审计阶段主动取消；尚未写出原始因子或 bootstrap 产物 |
| 元数据修正 | 修正继承的单方向 scope／路径标签，删除本轮未使用的 self／插值标签，避免三方向结果误标为仅正 z |
| 正式作业 | `3945255`，约 22:23 提交，单节点 8 CPU 核、48 GB 内存、上限 8 小时，候选分区为 cpu6248R、cpueicc、i72c512g |
| 正式执行 | 在 `cpu001` 运行，完成共同输入审计、逐方向 C3、配对 bootstrap、两个顺序、拟合和画图 |
| 完成状态 | `COMPLETED`，退出码 `0:0`，运行 4 分 16 秒；末条正式结果日志时间为 2026-11-07 22:29:09 |
| 补充检查 | 方向分辨检查图于同日完成，不改动正式拟合数据；原始结果和取消作业日志均保留 |

提交和阶段记录分别见 `BASE/submission.md`、`BASE/logs/F32_xyz_ratio_order_3945255.out`。

## 7. 数值与实现验收

### 7.1 收缩、索引与来源

1. 输入文件形状、有限性、三方向共同配置及 helicity 局域奇零点通过检查。
2. 新旧配置清单与完整 3000 行 bootstrap 索引通过逐项相等检查。
3. 重算正 z 的 C3 中心值、全部 bootstrap 及 C2 与原测试通过数值比较。
4. FFT 收缩在选定源汇距、插入时间、全部位移及样本上，与显式源时间求和独立抽查。
5. 对真实数据的四个通道／位移／时间检查点，分别取中心样本及两个 bootstrap，
   独立重算两个方向平均公式，并检验第 4.3 节的精确差异恒等式，均通过。
6. 非物理时间 padding、bootstrap 轴、helicity 原点零约束、CSV、PNG／PDF 和完成回执通过检查。

FFT 与显式源时间求和的最大绝对差如下；它们是数值实现残差，不是物理误差条：

| 方向 | unpolarized | helicity |
|---|---|---|
| x | \(8.89046\times10^{-18}\) | \(7.54405\times10^{-22}\) |
| y | \(1.44199\times10^{-17}\) | \(8.00742\times10^{-22}\) |
| z | \(1.00831\times10^{-17}\) | \(4.78394\times10^{-22}\) |

对应回执为 `BASE/results/contraction_audit_x.json`、
`BASE/results/contraction_audit_y.json`、`BASE/results/contraction_audit_z.json`。
本次文档整理再次核对 `complete.json` 登记的源码与输出 SHA256。
完成状态表示流水线执行和审计通过，不代表每个位移的常数拟合均通过质量门。

### 7.2 方向分辨信号检查

额外固定 \(T/a=10,\tau/a=5\) 画出三个方向的插入时间切片随位移变化。
在检查的 \(n=1,5,10\) 上，三个方向 U 中心值均为正，H 均为负，没有依据数据翻转符号。
这是既定正轴约定下的信号诊断，不是独立自旋／动量反转校准，也不说明所有方向间
幅度差异已经通过完整拟合检验。

## 8. 裸矩阵元比较结果

### 8.1 必须使用配对差值误差

两个方案共享配置和样本，不能按独立测量的误差相加来判断差异。定义

\[
\Delta^{(b)}(n)=M_B^{(b)}(n)-M_A^{(b)}(n),\qquad
\sigma_\Delta(n)=\operatorname{sd}_b\bigl(\Delta^{(b)}(n)\bigr),
\qquad p(n)=\frac{M_B(n)-M_A(n)}{\sigma_\Delta(n)}.
\]

正式汇总只取两方案均通过质量门的位移。相对差定义为
\(|M_B-M_A|/|M_A|\)，误差比为 \(\sigma_B/\sigma_A\)。

| 通道 | 两方法共同通过 | 最大绝对配对 pull | 中心值相对差中位数 | B/A 误差比中位数 |
|---|---|---|---|---|
| unpolarized | 18/24；\(n=6,\ldots,23\) | \(0.23852\) | \(0.4120\%\) | \(1.000861\) |
| helicity | 20/23 个非零位移；\(n=1,2,6,\ldots,23\) | \(0.52006\) | \(0.2267\%\) | \(1.001163\) |

两通道均没有共同通过点的绝对配对 pull 超过 2。B/A 误差比的中位数只比 1 高约
\(0.086\%\) 和 \(0.116\%\)，目前没有实质性统计优势。

### 8.2 部分正式通过点的数值

| 通道、\(n=z/a\) | A：先平均再 ratio | B：先 ratio 再平均 | 配对 pull |
|---|---|---|---|
| U，6 | \(0.406988\pm0.014241\) | \(0.407063\pm0.014173\) | \(0.05487\) |
| U，10 | \(0.095641\pm0.007641\) | \(0.095640\pm0.007643\) | \(-0.00138\) |
| U，23 | \(0.003714\pm0.004563\) | \(0.003750\pm0.004569\) | \(0.22614\) |
| H，1 | \(-0.017721\pm0.001620\) | \(-0.017729\pm0.001624\) | \(-0.05161\) |
| H，10 | \(-0.012045\pm0.002249\) | \(-0.012060\pm0.002252\) | \(-0.14124\) |
| H，23 | \(0.001879\pm0.001433\) | \(0.001916\pm0.001436\) | \(0.52006\) |

误差和差值统计均来自配对 bootstrap，而不是由此表四舍五入后的值重新推算。

### 8.3 未通过点与不能推出的结论

- U 的 \(n=0,\ldots,5\) 在两方法中都只作 fallback／诊断。
- H 的 \(n=3,4,5\) 在两方法中都只作 fallback／诊断。
- H 的零点是精确奇对称约束，不作为有统计误差的拟合点，故正式分母是 23 个非零位移。
- 图中实心表示拟合通过，空心表示诊断；精确零点以灰色菱形表示。

方向平均没有自动解决这些位移的残余时间依赖或模型／统计匹配问题。
不能因为两个顺序很接近，就把两者都未通过的点认定为可靠矩阵元；
也不能仅凭常数拟合失败就把原因唯一归于某一种激发态机制。

## 9. 图像、数据结构与复现入口

| 产物 | 内容 |
|---|---|
| `BASE/run.py:70` | 两种平均顺序的数值实现 |
| `BASE/run.py:97` | 方向轮换、OPE／2pt 读取和投影输入 |
| `BASE/run.py:184` | 共享索引下的逐方向 C3、分母及两个 ratio |
| `BASE/run.py:273` | 共同候选拟合及配对差值 |
| `BASE/run.py:308` | 比较图、CSV 和完成审计 |
| `BASE/diagnose_directions.py:20` | 固定方向分辨信号检查 |
| `BASE/results/bootstrap_indices.npz` | 唯一重采样索引和配置顺序 |
| `BASE/results/c3_x.npz`、`c3_y.npz`、`c3_z.npz` | 方向分辨 C3、C2 和分母状态；样本轴第一项为中心值，其余 3000 项为 bootstrap |
| `BASE/results/ratios.npz` | 两方案两通道的中心值、全部 bootstrap 及共同有效网格 |
| `BASE/results/fits_unpolarized.npz`、`fits_helicity.npz` | 完整候选、质量、AIC 权重、裸矩阵元及配对差值／相关系数／pull |
| `BASE/results/fit_values.csv` | 逐位移中心值、误差、质量和 fallback 比例 |
| `BASE/results/complete.json` | 正式完成回执和源码／输出哈希 |
| `BASE/results/direction_checkpoint.done.json` | 方向分辨检查的独立回执 |

`ratios.npz` 的中心值轴为

`method, channel, z, tsep, tau`，

bootstrap 轴为

`method, channel, boot, z, tsep, tau`。

两个方法的顺序固定为 `average_then_ratio`、`ratio_then_average`；
通道顺序为 `unpolarized`、`helicity`。不能在比较时误置 bootstrap 与位移轴。

图像均在 `BASE/results/plots`：

- `unpolarized_bare_matrix_order_comparison.png` 及同名 PDF。
- `helicity_bare_matrix_order_comparison.png` 及同名 PDF。
- `unpolarized_paired_order_pull.png`、`helicity_paired_order_pull.png`。
- `unpolarized_direction_checkpoint.png`、`helicity_direction_checkpoint.png` 及同名 PDF。

裸矩阵元比较图只保留单面板，误差棒不连线、两个方案横向微错位，pull 单独保存。
完整结果说明与提交历史分别见 `BASE/RESULTS.md`、`BASE/submission.md`。

重放入口如下；原产物存在时程序会校验后复用，来源哈希变化时拒绝静默覆盖。
若要改变科学设置，应另建版本，而不是修改现有结果对应的生产脚本。

```bash
cd /public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/F32P30_DirectionOrder_20261107
/public/home/donghx/AnaConda/bin/python -B run.py validate
```

完整流程使用 `run.py all`；各阶段也可单独运行 `prepare`、`bootstrap`、`fit`、`report`。

## 10. 当前结论与后续检查

**结论：在 F32P30 的三个正轴、共同 777 个配置、3000 次配对 bootstrap 和当前
相关常数模型下，没有发现 ratio 与方向 average 的顺序引起显著的逐点差异，
两者统计误差也几乎相同。当前测试没有给出明显偏好其中一种顺序的统计依据。**

这不等于证明两个非线性估计量严格相同，也不构成全位移协方差的整体一致性检验。
本轮没有完成负动量方向的反转核验、其他动量／ensemble 的重复测试、
激发态系统误差控制、分箱自相关分析或 helicity 独立绝对相位校准。

后续需要保持共同配置和样本索引，分别检查模型／窗口稳定性、负方向的 spin/link
约定以及不同动量的比值偏差；不要因某点误差更小就逐点选择平均顺序。
若扩展到 self-ratio，必须另行声明方向参考分母和裸幅度恢复方案，因为

\[
\left(\sum_dw_dS_d\right)\left(\sum_dw_dA_d\right)
\ne\sum_dw_dS_dA_d
\]

一般成立，不能用“平均 self”乘“平均锚点”冒充各方向裸 ratio 的平均。
ratio／self-ratio 的独立测试和零距离归一化讨论仍保存在
`/public/home/donghx/.codex/Markdown/self_ratio.md`；本文件专门记录方向平均顺序测试。
