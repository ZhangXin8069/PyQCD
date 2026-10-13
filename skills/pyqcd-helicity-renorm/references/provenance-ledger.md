# Helicity/EMT provenance ledger

## 1. 状态

来源提交：`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。

```text
invocation: explicit-only
status: NOT_PRODUCTION
literal coefficient evidence: E0
selected placement/ODE checks: E1
```

内置 `EMT_MatchingCoeff`、两类 `Helicity_MatchingCoeff`、entrywise Padé 和
`Helicity_RunningFactorCalculator` 的 decimal tables 均不能作为 production QCD
coefficient，除非 caller 提供独立核实的矩阵与完整 basis。

## 2. Literal matching contract

```text
(R11,R12,R21,R22) -> R = [[R11,R12],[R21,R22]]
a = alpha_s/(4*pi)
L = ln(mu^2/muR^2)
series through a^3
```

equal-scale 局部检查：

```text
R12 = 8a + O(a^2)
R21 = -6a + O(a^2)
```

这只检测 row/column swap，不识别物理 operator basis。

`Helicity_MatchingCoeff` 的 rational `R11` branch 与
`Helicity_MatchingCoeff_tmp` 不同；其 resummation 关系未映射。

## 3. Padé 与 loop semantics

```text
loop=1,2,3:
  explicit local coefficients

loop=4, is_pade=0:
  fourth-order coefficient = 0
  therefore explicit three-loop truncation

loop=4, is_pade=1:
  r4_ij = (r3_ij)^2 / r2_ij
```

entrywise estimate 不是四圈计算。denominator near zero 必须 fail closed；Eq. (25)
的 large-beta0 observation 不是该 Padé rule。

## 4. Beta source

Baikov--Chetyrkin--Kühn, arXiv:`1606.08659v2`, p. 1, Eqs. (1)--(5):

```text
a_s = alpha_s/pi
d a_s/d ln(mu^2) = -sum_i beta_i a_s^(i+2)
```

它支持 five-loop coupling context，不识别 helicity/EMT matching、ordered
two-operator vector 或 `Gamma` entries。

Default `Lambda=0.332 GeV` 只适用于项目约定的 `nf=3`；其他 flavor 必须显式提供
匹配的 `Lambda`。

## 5. Primary candidate conflict

Zhao et al., arXiv:`2512.24315v2`, Supplemental pp. 8--10,
Eqs. (16), (18), (20)--(25):

```text
ordered vector: (Delta G, Delta Sigma)^T
operators:      (K^mu, J5^mu)
source:         RI/MOM
target:         modified MSbar (Larin)
gauge:          Landau for printed finite terms
Nc:             3
color trace:    1/2
a_s:            alpha_s/(4*pi)
L_mu:           ln(mu^2/mu_RI^2)
```

Printed source:

```text
R12 = 1 + a_s C_F (3 L_mu + 6) + ...
R21 = 1 - 2 nf a_s + ...
```

Legacy equal-scale values:

```text
R12 = 8a + ...
R21 = -6a + ...
```

For `nf=Nc=3` and `a=alpha_s/(4*pi)`, each source tree term differs from the legacy
return by exactly one. Eqs. (20)--(21) term-match `tmp` `R11/R22` fixed-order
pieces, but that does not repair the off-diagonal conflict or bind the basis.

## 6. Running table conflict

Source Eq. (24) gives the left action and relations:

```text
dR/dln(mu^2) = Gamma R
gamma11 = -beta
gamma21 = 0
gamma22 = -2 a_s nf gamma12
```

arXiv:`2201.01797v3`, Eq. (10) gives an intended `gamma12` chain. After the
documented factor `-4` `K^mu` normalization and conversion to `alpha_s/pi`, the
printed chain contains:

```text
4, 76, 316.679353...
```

Legacy source rounds the last value to `316.679`, rounds `643.833` for the beta
numerator instead of `3863/6`, never declares the paper basis, and omits the
related quartic `gamma22` contribution. Status:
`PRIMARY_RELATION_PARTIAL_TRUNCATION_CONFLICT`.

## 7. Refusal dispositions

| Situation | Required disposition |
|---|---|
| missing basis/scheme/gauge/equation | `INCOMPLETE_PHYSICAL_PROFILE` |
| bind Zhao v2 matching to legacy API | `PRIMARY_SOURCE_CONTRADICTION` |
| running basis/RG map incomplete | `UNVERIFIED_PHYSICAL_PROFILE` |
| complete-looking but unregistered metadata | `UNVERIFIED_NO_MAPPED_PHYSICAL_PROFILE` |
| ordinary literal call | `UNVERIFIED_LEGACY`; not production |

Even a successful ODE means only that the numerical integration succeeded. It does not
validate `Gamma`, the matrix orientation, the operator basis or the gauge.
