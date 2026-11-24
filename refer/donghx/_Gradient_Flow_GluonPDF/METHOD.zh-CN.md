# m0：固定 0.1–0.4 fm 窗口，使用新 cut4 裸量

## 本次问题与预注册

日期2026-11-22。接续上一轮 `0.8<T*a<1.1 fm, cut=4` 裸量检测，本次主输入为新cut4的常数/AIC裸拟合结果。为避免把改变z范围与改变裸拟合混淆，同时处理相同物理z窗口的旧cut2 AIC和cut4固定全T两条对照。三条路线分别计算，不合并、不按Q择优、不覆盖旧产品。

用户要求的窗口解释为闭区间 `0.1 <= z_fm <= 0.4`，纳入全部实测格点，不插值、不纳入z0。实际点如下：

|ensemble|Nconf|a [fm]|z/a|z [fm]|单参数模型自由度（无删模）|
|---|---:|---:|---|---|---:|
|C24P29|120|0.105|1,2,3|0.105,0.210,0.315|2|
|E32P29|100|0.0897|2,3,4|0.1794,0.2691,0.3588|2|
|F32P30|70|0.0775|2,3,4,5|0.1550,0.2325,0.3100,0.3875|3|

保留原配置bootstrap全部500行、原顺序、组态和索引；不对拟合值二次bootstrap，不因坏行而裁剪。仅处理unpolarized connected u-d、gamma4、Pz=0、单空间z方向的实部。P0的虚部由对称性精确为0，不作为拟合观测。

## 从新裸量到m0输入

在本次独立目录对上述短z范围执行下列链，而不是直接拿新裸量与MSbar短距离系数相除：

1. 使用先前H3的fundamental系数d，对裸量只乘一次 `exp(+d*z/rf)`；不再加入另一个重叠的微扰线质量指数。C24/E32/F32的d为0.38751363314231924、0.3691984652781599、0.34232525660315893，原JK误差单列。
2. 同一flow、同一原数据中心或同一bootstrap行，以z0局域裸量归一化；局域点保持精确1，不作为m0数据。
3. 复用上轮已冻结且核验过的局域归一化GF转换系数，通过固定阶NLO和NNLO分别转换，mu=2 GeV、nf=3、alpha_s=0.29812517844614284。无RGR、无LaMET，也没有把ordinary flowed fields擅称ringed。
4. 延续已声明的 `rf>0.15 fm` 选择：tau64=0.4,0.5,0.6,0.7,0.8，每个ensemble五个相同物理flow，rf约0.16046,0.17940,0.19652,0.21227,0.22693 fm。`tf=tau64*(0.0897 fm)^2`，不是把相同tf/a²当成相同物理时间。
5. 每个z、每个阶数，以五个flow的完整配对协方差拟合 `H(tf)=H0+slope*tf_fm2`。五点全窗固定为主结果；两个连续四flow子窗仅作敏感性，不按较小chi2替换主截距。这里只是固定a下线性零flow截距诊断，不是已控制的连续零flow极限。

三条裸量路线为：主 `cut4_AIC`，对照 `cut2_AIC` 和 `cut4_fullT`。后者直接使用之前已保存的cut4全T候选重构结果，不再拟合裸量或增加参数；失败全T值保留诊断，绝不提升为质量通过。

## m0模型、短距离目标与符号

保持上一轮相同定义：

\[
 H_0(z,P=0)=C_0^{\overline{\mathrm{MS}}}(z,\mu)
 \exp[-m_0 z_{\rm fm}/(\hbar c)]+\text{scheme/power/cutoff remainders},
 \qquad \hbar c=0.1973269804\ {\rm GeV\,fm}.
\]

拟合 `log(H0/C0)=-m0*z_fm/hbarc`，一个参数m0[GeV]，截距固定0，不加z²项或自由归一化。m0不是夸克质量，也不是无条件物理线质量。若未来施加有限质量变换，对应乘 `exp(+m0*z/hbarc)`；本次不自动应用。

目标C0不是GF匹配多项式，而是与空间Wilson线正交的gamma4短距离零Ioffe-time系数。直接复用上一轮 `quark_gf_adj75_boot500_rfgt015_m0_20261122/code/fit_core.py` 的 `target`，来源及推导记录于其 `METHOD.zh-CN.md` 和已存原始文献arXiv:2006.12370v3 Eqs.(22)–(25)：C0=iA(omega=0)/2，xi4=0排除另一不变振幅，指数差积分在omega=0消失。

令 `L=log((z_fm/hbarc)^2*mu^2/4)+2*gamma_E`，则

\[
 C_0^{\mathrm{NLO}}=1+\frac{\alpha_s C_F}{2\pi}\left(\frac32L+\frac52\right),
\]
\[
 C_0^{\mathrm{NNLO}}=C_0^{\mathrm{NLO}}+\frac{\alpha_s^2}{2\pi^2}
 \left\{
 C_F^2\left[\frac9{16}L^2+\left(\frac{\pi^2}{3}+\frac{25}{16}\right)L-4\zeta_3+\frac{\pi^2}{9}+\frac{223}{192}\right]
 +C_FC_A\left[\frac{11}{16}L^2+\left(-\frac{\pi^2}{12}+\frac{53}{16}\right)L+\zeta_3-\frac{5\pi^2}{24}+\frac{4877}{576}\right]
 +n_fC_FT_F\left[-\frac14L^2-\frac54L-\frac{469}{144}\right]
 \right\}.
\]

CF=4/3、CA=3、TF=1/2。NNLO前因子为alpha_s²/(2*pi²)，不是误少一倍的(alpha_s/(2*pi))²。NLO输入配NLO目标，NNLO输入配NNLO目标；有限阶C0直接取对数，不进行对数的再次级数展开。

## 统计、质量与响应

所有对数要求所选窗口中原数据中心和全部500行的H0及C0严格正且有限；不取绝对值、不裁剪、不删除负行。使用同500行的完整跨z对数协方差，ddof=1。中心是原数据中心的处理结果，而非bootstrap均值。

flow与m0都使用相关矩阵eigenmode deletion：主cut=1e-8，样本数值秩阈值1e-12；1e-10/1e-6作敏感性。禁止以对角协方差替换、任意抬大误差或依据Q事后改阈值。m0的dof=保留协方差秩−1。参数误差为全部500行的SD，不除sqrt(500)。质量门Q>=0.05、chi2/dof<=3，统计门与物理输入资格分别保存。

M(z)/M0中局域M0在同一bootstrap行代数消去，故本次另外构建去M0后的adjacent链质量及局域分母安全门；原绝对裸量质量同时保存。源裸拟合通过、flow线性拟合通过、m0线性模型通过和尺度/H3训练域通过是四种不同状态，不能互相替代。短距有限flow、a/z、固定a外推、H3仅部分flow训练及NP/PT有限方案未闭合的问题全部保留。

将d±原JK误差、alpha输入±及四圈beta响应通过同一名义flow权重传播，再以同一名义m0权重计算质量响应；alpha变化同时改变GF系数和C0。不将这些中心响应当作独立随机行与矩阵元bootstrap配对或平方合并成总误差。窗口、阶数与协方差变化单列。对route之间差值直接使用同500行配对差SD，pull只作描述性敏感性，非全局显著性。

枚举本次0.1–0.4 fm中全部至少2点的连续z子窗仅作敏感性，总计24个主route窗口；三route合计72个。flow窗口变化与flow协方差SVD变化也传播到完整物理z窗的m0。没有“最佳Q”自动选主窗。

## 核验与来源

所有数组运算、测试、拟合、绘图在CPU Slurm。主计算源码及206份源文件/产品在提交前冻结；核验与绘图另有冻结清单。原有已提交代码和所有上游产品均未改写。

- 保留旧cut2一路独立重算，逐行复现已保存的GF匹配及零flow结果，验证此次改变窗口没有偷换估计量。
- 后处理GF系数用数值多项式独立检查；短距离C0用另一套逐色因子表达式核验，不把同一目标函数再调用一次当独立验证。
- flow采用独立precision/information普通相关GLS验证，与生产whitening/lstsq解比较。
- m0采用独立协方差逆/删模precision验证参数、500行、chi2、dof、Q、权重、SVD及输入响应。
- 14项继承边界测试加五flow直线真值、三个物理z mask检查；未因拟合失败而删样本或升格物理结论。

入口数据：`primary.csv`、`primary_all_routes.csv`、`mass_fit_windows.csv`、`flow_fit_windows.csv`、`paired_route_comparison.csv`、`flow_window_mass_sensitivity.csv`、`flow_svd_mass_sensitivity.csv`。每ensemble的 `processed_P0.npz` 保存从新裸量到各阶GF数据及输入响应；每route/order的 `flow_fits.npz` 和 `mass_fits.npz` 保存完整500行、协方差、窗口、权重及参数。图在 `plots/`，原始来源/索引/哈希由 `source_lock.json`、`input_manifest.json` 及收据追溯。
