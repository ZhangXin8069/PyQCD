---
name: pyqcd-helicity-renorm
description: |
  Use when explicitly auditing legacy helicity or EMT matching, R12/R21
  placement, nf=3 coupled running, or Gamma truncation against primary sources.
---

# pyqcd-helicity-renorm — explicit audit-only 技能

## 目的与边界

本技能用于审查 legacy `2 x 2` helicity/EMT matching 与 coupled running，
而不是发布 production coefficient。

- 仅允许显式调用；普通出现“helicity”一词不应自动加载本技能。
- 可以审查 tuple placement、input domain、ODE orientation 和来源冲突。
- 可以使用 caller-supplied、已核实 basis 的 coefficient matrices。
- 内置 decimal matching 与 anomalous-dimension tables 未完成 equation-level map，
  因此 `not production`。

## 适用与不适用

| 适用 | 不适用或必须转交 |
|---|---|
| 审查 `R=(R11,R12;R21,R22)` 的 flat order | 把 tuple 位置解释为未声明的 quark/gluon basis |
| 核对 `R12/R21` 与 primary printed equation | 用近似项相同掩盖 tree-level 冲突 |
| 检查 `loop=4,is_pade` 与 `Gamma` 截断 | 把 entrywise Padé 称为四圈计算 |
| 对 caller-supplied matrices 检查 RG 方向与 composition | 把内置 decimals 称为已验证物理 |
| 输出 provenance refusal/audit | bare RI vertex、gauge-field volume work 或最终 TMD |

## 核心公式与归一

内置 matching 使用

$$
a=\frac{\alpha_s(\mu)}{4\pi},\qquad
L=\ln\frac{\mu^2}{\mu_R^2},
$$

返回顺序固定为

```text
(R11, R12, R21, R22)
```

但该顺序只固定数组位置，不声明 `(O1,O2)` 的物理 identity，也不声明 matrix action
是 `O_out=R O_in`、转置还是 inverse。

来源中的五圈 beta 使用

$$
a_s=\frac{\alpha_s}{\pi},\qquad
\frac{da_s}{d\ln\mu^2}
=-\sum_{n=0}^{4}\beta_n a_s^{n+2}.
$$

coupled running 采用左作用

$$
\frac{dR}{d\ln\mu^2}=\Gamma(a_s)R,\qquad
U(\mu_2,\mu_1)=R(\mu_2)R(\mu_1)^{-1}.
$$

内置 equal-scale 一阶 off-diagonal 字面值为

$$
R_{12}=8a+O(a^2),\qquad
R_{21}=-6a+O(a^2).
$$

它们只用于检测 `R12/R21` swap。`loop=4,is_pade=0` 实际仍是显式三圈；
`is_pade=1` 使用

$$
r_4^{ij}=\frac{(r_3^{ij})^2}{r_2^{ij}},
$$

这是 entrywise estimate，不是四圈 coefficient。

## 强制来源门

Zhao v2 的 `(Delta G, Delta Sigma)^T` / `(K^mu,J5^mu)` RI/MOM-to-modified
`MS-bar` Larin 问题给出

$$
R_{12}=1+a_sC_F(3L_\mu+6)+\cdots,\qquad
R_{21}=1-2n_fa_s+\cdots .
$$

legacy matching 的对应 tree terms 为 0，因此 `R12/R21` 存在确定源冲突。
不能用后续阶项相似、`R11/R22` term match 或 rational branch 近似来抹平。

`Gamma` 结构还要求核对

$$
\gamma_{11}=-\beta,\qquad
\gamma_{21}=0,\qquad
\gamma_{22}=-2a_sn_f\gamma_{12},
$$

但 legacy table 使用 rounded `643.833`、`316.679`，并在保留三阶
`gamma12` 的同时漏掉下一相关 `gamma22` 项。任何 ODE solver success 都不能修复
coefficient/basis provenance。

## 工作流

1. 确认调用是 explicit；若请求默认 production matching/running，先拒绝物理解释。
2. 要求 ordered operator vector、source/target scheme、gauge、`nf/Nc`、color trace、
   operator normalization、matrix action 和 direction。
3. 对每个 `Rij` 与 `Gammaij` 建立 paper version/page/equation；缺一项即停止。
4. 核对 coupling variable：`alpha_s/pi` 与 `alpha_s/(4*pi)` 不可混用。
5. 明确 `loop`、Padé 或 explicit truncation，保存 denominator 与 scale domain。
6. 只对 verified caller-supplied matrices执行 ODE；检查 equal-scale identity、
   composition residual 和 matrix orientation。
7. 输出 audit record，状态保持 `UNVERIFIED_LEGACY` 或
   `PRIMARY_SOURCE_CONTRADICTION`，不生成 production number。

## 验收门

- [ ] 仅显式调用；策略允许 `allow_implicit_invocation: false`。
- [ ] `(O1,O2)`、row/column 与 left/right action 已声明。
- [ ] `R12/R21` 的 source conflict 保留在结果和报告中。
- [ ] `Gamma` 的四项、rounding、truncation 与 source equation 已核对。
- [ ] `loop=4,is_pade=0/1` 未被误标为四圈。
- [ ] `nf != 3` 不复用 legacy `Lambda` 或 anomalous dimensions。
- [ ] ODE 使用 verified matrix；solver 收敛不等于 coefficient 已验证。
- [ ] 协方差、scale variation 和 truncation uncertainty 不合并为单一无标签误差。
- [ ] 最终状态明确为 `not production`，除非未来逐项修复并独立复核。

## 与 PyQCD 技能交接

| 技能 | 交接责任 |
|---|---|
| `pyqcd-gauge` | 提供规范作用量、gauge fixing 和 lattice observable context；不能以 helicity 名称替代 gauge provenance。 |
| `pyqcd-tmd-algorithm` | 不接收内置 legacy matching/running；只有 caller-verified basis 与 exact equations 才可作为外部输入，并按 TMD matching 阶段单独验证。 |
| `pyqcd-statistics` | 对 caller-supplied resamples 执行协方差、SVD/秩门和 scale/truncation 系统误差分解；不得把 ODE 输出当统计证据。 |
| `pyqcd-analysis` | 报告和图表必须显示 `not production`、source contradiction 与未闭合 truncation；不得只展示 fitting curve。 |

## 来源与证据上限

来源技能冻结于
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。逐项冲突与 refusal ledger 见
[`references/provenance-ledger.md`](references/provenance-ledger.md)。

五圈 beta context 来源为 arXiv:1606.08659v2, p. 1, Eqs. (1)--(5)。
helicity matching candidate 为 arXiv:2512.24315v2 Supplemental pp. 8--10,
Eqs. (16), (18), (20)--(25)；其 printed tree terms 与 legacy返回值冲突。
相关 singlet-axial 系数链见 arXiv:2201.01797v3, Eq. (10)，但本库未绑定相同 basis。

当前证据上限：E0 的 literal coefficient contract；仅 tuple placement、部分一阶
off-diagonal 与 ODE orientation 有局部 E1 检查。没有 production matching/running、
interacting observable、完整 truncation uncertainty 或独立 physical profile。
