# Bare matrix fit contract

## 1. 估计量

对 insertion time `tau` 和 source-sink separation `T`，基础 ratio 记为
`R(T,tau)`。本技能只拟合已经声明并 mask 过的 ratio，不重新定义上游相除顺序。

可选模型：

### Direct Ratio

\[
R(T,\tau)=M.
\]

适合作为交叉检查。未独立约束 C2 gap 时，默认不能用自由 one-gap Ratio 模型。

### Undifferenced Sumratio

\[
S_{L,R}(T)=\sum_{\tau=L}^{T-R}R(T,\tau)
=B_{L,R}+M N_T,\qquad
N_T=T-L-R+1.
\]

保留 intercept，因而与直接 Ratio 的 covariance 和 excited-state response 不同。

### Forward Sumratio-difference

\[
S_c(T)=\sum_{\tau=c}^{T-c}R(T,\tau),
\qquad
D_c(T)=S_c(T+1)-S_c(T)=M.
\]

difference 由 lower endpoint `c` 标记。`D(Tmin)` 消费 `S(Tmin)` 和 `S(Tmin+1)`，
不能误读为 `S(Tmin-1)`。

## 2. Physical Tmin

每个分析注册一个物理阈值，例如 `tmin_fm_threshold=0.6`：

\[
Tmin=\lfloor tmin\_fm\_threshold/a\_fm\rfloor+1,
\qquad
Tmin\,a\_fm>tmin\_fm\_threshold.
\]

参考 ensemble 示例：

| ensemble | a [fm] | Tmin | Tmin*a [fm] |
|---|---:|---:|---:|
| L24x72 | 0.1050 | 6 | 0.6300 |
| L32x64 | 0.0897 | 7 | 0.6279 |
| L32x96 | 0.0775 | 8 | 0.6200 |

`Tmin`、`a_fm`、阈值、窗口和单位必须随每个输出保存。候选网格全局注册；不能因为
某个 `z/p/flavor/channel` 在更短 separation 上看起来更好而降低 `Tmin`。

## 3. Direction/projection contract

- 明确方向平均发生在构造 `S/D` 前、direct `R` fit 前还是独立 fit 后。
- 默认在同一 shared bootstrap replica 内先平均 ratio values，再拟合。
- joint fit 时保留 cross-direction covariance。
- 六个方向必须共享组态和 index；negative-momentum odd 方向先乘声明过的 parity
  因子再作符号比较。
- 记录复数投影、even/odd、整体符号和代数零位置。`~1e-17` 的零点不能升级成高显著度
  物理 tension。
- gluon tensor 组合必须在输出中显式写清，例如 `T=ti+tj`、`S=2ij`、
  `T-S=ti+tj-2ij`；未包含两个 orientation 的 plus-z-only 产品不能标成 even-z 结果。

## 4. Fit gates

默认注册门：

```text
finite inputs and valid mask
identifiable covariance/design rank
positive dof
Q >= q_min
correlated chi2/dof <= chi2_limit
finite bootstrap fraction >= configured minimum
```

参考值常用 `Q>=0.01`、`chi2/dof<=3`、finite fraction `>=0.90`；它们是注册门，
不是跨项目普适常数。检查 raw sample rank、effective rank、condition number、
shrinkage、eigenvalues、SVD cut、dof 和 failure reason。

## 5. Same-cut AIC

固定 cut 后，对每个 candidate window `i` 和 replica `b`：

\[
AIC_{i,b}=\chi^2_{i,b}+2k+2N_{omit,i},
\]

其中 `k` 是 fit parameter 数。权重为

\[
w_{i,b}=
\frac{e^{-\Delta AIC_{i,b}/2}}
{\sum_j e^{-\Delta AIC_{j,b}/2}}.
\]

omission penalty 使不同长度 nested range 可在同一 cut 内比较。不要用不同 data-vector
长度的 Gaussian log-determinant likelihood 直接比较；它可能给短窗几乎单位权重。

不同 cut 的 summed observable 和 likelihood normalization 不同：

1. 先在各 cut 内 AIC-average windows。
2. 再把 usable cuts 按注册政策混合；参考实现使用 equal-weight cut mean。
3. 保存 usable cut count、cut-to-cut spread 和 `single_cut` 状态。

## 6. Complete bootstrap selection

对每个固定 `(model,channel,momentum,z,replica)`：

1. 对通过该 replica quality gates 的所有 candidate 作 AIC average。
2. 若没有 candidate 通过，选择有限 rejected candidate 中 **precomputed candidate
   statistical error** 最小者。该误差是该 candidate 在整个 finite bootstrap ensemble
   上的 `std(M_boot,ddof=1)`，不是从单 replica 临时制造。
3. tie-break 依次用较小 raw `chi2`、注册 candidate 顺序。
4. 若无 finite statistical error，选 finite candidate 中 raw `chi2` 最小者。
5. 若无 positive-dof candidate，只能使用明确注册的 single-point/zero-dof
   diagnostic；不能把任意单点当正常拟合。
6. 若没有 finite candidate，保留 NaN 和 `no_finite_candidate`。

参考实现会逐 replica 重新选择 candidate，因此最终 `std(M_boot_selected,ddof=1)`
已经包含 replica-dependent window/fallback 的 spread。不要再除 `sqrt(Nboot)`。

## 7. Accepted 与 filled 字段

必须分开保存：

```text
M_quality, M_boot_quality, quality_accepted
M_selected, M_boot_selected, selected_available
fallback_used, fallback_status, fallback_candidate_index
fallback_fraction, candidate_statistical_error
```

典型 v2 产品字段：

```text
M, M_boot, statistical_error, fit_accepted/fit_valid
M_filled, M_boot_filled, filled_statistical_error
estimate_available
central_fallback_used, central_fallback_candidate_index,
central_fallback_status
replica_fallback_used, replica_fallback_fraction,
replica_minchi_candidate_index
```

filled coverage 只保证矩形数组可写，不保证质量。fallback fraction 接近 1 时，下游
必须把该点和主要统计误差标成 fallback-driven。

## 8. Ratio/Sumratio paired comparison

若两个方法消费同一底层数据和相同 index，定义

\[
p=\frac{M_R-M_S}{\operatorname{sd}_b(M_{R,b}-M_{S,b})}.
\]

不能将两个独立误差 quadrature 后作 pull。要求同一 projection、cut、方向平均阶段、
物理 `Tmin`、 admissible separations 和 mask。若方法不一致：

- 不平均掉差异；
- 不选更小误差的方法；
- 标记 unresolved excited-state/window dependence；
- 尝试 registered later `Tmin` 或独立 C2-constrained multi-state model。

## 9. Excited-state boundary

直接 Ratio 的 leading structure 是

\[
R(T,\tau)=M+Ae^{-\Delta E\tau}
+Be^{-\Delta E(T-\tau)}+Ce^{-\Delta ET}+\cdots .
\]

自由拟合 `DeltaE` 不一定可辨识。只有独立 C2 spectrum 提供可靠 gap 时，才能注册
shared/constrained multi-state diagnostic。优化器收敛不能证明 data identifiability。

## 10. 输入/输出身份

最低输入字段：

```text
estimator_id
configuration manifest + hash
bootstrap method, Nboot, resample size, seed and exact indices
channel, flavor, pabs, direction, even/odd, z, tsep, tau
valid mask and complex projection
operator convention and overall sign
input hashes and upstream code version
```

最低输出字段：

```text
central/accepted and bootstrap arrays
statistical error and finite fraction
quality gate table and failure reasons
candidate values, chi2, Q, dof, covariance diagnostics
AIC weights and omitted counts
cut/window spread and method comparison if requested
configuration/index identity, hashes, command and status
```
