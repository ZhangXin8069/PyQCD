# TMD renormalization reference — disconnected matrix, CS, matching, limits

## 适用范围

任务进入核子外态、disconnected 真空扣除、软/快度方案、自重整化、Fourier、Collins–
Soper 核、NLO 匹配或连续极限时读取。本文件承接 `geometry.md` 已通过的算符对象。

## 核子外态与 disconnected

对每个组态保留核子两点 `C2`、独立胶子 loop `Lg` 和逐组态积：

\[
C_3^g=\langle C_2L_g\rangle-\langle C_2\rangle\langle L_g\rangle,
\qquad R_g=C_3^g/C_2.
\]

`run_disconnected_tmd_ratio` 的输入形状可以是
`ope_all[cid]['tmd']=(nz,nb,Nt)`，但 `C2*OPE` 只有在 OPE 确实是同组态独立测得的
loop 时才具有该因子化解释。真空扣除必须在重采样后的 ensemble 均值上完成；`Ncfg=1`
时逐组态扣除会恒等为零。没有多个 `t_sep`、独立 loop 来源和平台/多态稳定性时，
`c0` 只能称为数值骨架。

## 软因子、自重整化与混合

方案模板可写成

\[
H_{g,A}^{\rm sub}(z,b;\tau,\ell)=
\frac{h_{g,A}^{\rm flow}(z,b;\tau,\ell)}{\sqrt{S_\tau^g(b,\ell)}}R_\tau^g.
\]

`S_tau` 必须和主算符共享流时间、表示、方向、转角、长度及归一化；`R_tau` 的短距
比值、自重整化或混合定义必须单独写出。可用接口包括 `build_hB_dataset`、
`boot_covariance`、`fit_ZR`、`fit_ZR_samples`、`fit_hR_lambda`，但它们只是 Z_R/混合
方案的数值骨架；没有同几何软因子和 rapidity 处理时不得称为完整 TMD 重整化。

## Fourier、CS 与匹配边界

| 物理步骤 | API | 必须说明 |
|---|---|---|
| cos 型准 TMD | `quasi_tmd_pdf` | `h_R(z,b)`、单位和偶性；不能单独证明 soft/rapidity 已消除 |
| sin 型共线准 PDF | `quasi_pdf_gluon` | `x→0` 保护；是 collinear 交叉检查，不是 TMD 替代 |
| CS 两动量比值 | `cs_kernel_two_momentum` | 至少两个 `Pz`，同一 `(z,b,ell,tau)`，记录 `z_ref` 与 `k_clip` |
| 一圈混合匹配 | `tmd_matching_hybrid` | 保留 `Z_ij`、`mu/zeta/alpha_s`、表示和阶数；输出仍可能是 scaffold |
| 流到 MS 系数 | `sftx_gluon_matching_coeff` | 只作局部流方案 building block |
| 联合外推 | `fit_hR_PDF_extrap_boot` | 需要多 `a/Pz/m_pi/L` 和协方差 |

匹配必须显式保留

\[
\boldsymbol C^g\otimes(f_g,f_q)^T,
\qquad Z_{ij}=\delta_{ij}+O(\alpha_s C_A/(2\pi)).
\]

检查 `alpha_s→0` 是否回到单位核；矩阵病态时报告条件数和截断，不能静默产生伪结果。
`soft_function_intrinsic` 的 `R/R[0]` 归一化和 `soft_factor=1.0` 都不能自动解释为
已测量软因子。

## 统计与连续极限

`z,b,ell,tau,±z,Pz` 必须共享组态索引，使用带记录的 block-jackknife/bootstrap、
协方差条件数和 SVD 参数。依次检查 `ell`、`tau→0`、`Pz→∞`、`a→0`、有限体积和拟合窗，
输出统计误差、系统误差、协方差和外推带。统计细节转 `pyqcd-statistics`，不要在本文件
重新定义重采样算法。
