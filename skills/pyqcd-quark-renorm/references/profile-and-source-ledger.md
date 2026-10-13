# Quark profile registry 与来源台账

## 1. 状态与纪律

来源提交：`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。

`MAPPED` 表示已核对 paper version、页码、方程、expansion variable 与
source-to-code algebra。它不证明 lattice projector、gauge-fixed NPR、
continuum extrapolation 或 uncertainty budget。

任何 profile 缺字段时，结果只能标为：

```text
INCOMPLETE_PROFILE
UNVERIFIED_LEGACY
LITERAL_REPLAY
```

禁止用相邻函数、相似 decimal 或通用论文为整张表背书。

## 2. Profile 必需 schema

```text
symbol
status: MAPPED | MAPPED_DERIVED | UNVERIFIED_LEGACY
paper, version, page, equation
source_scheme, target_scheme, gauge
operator_basis, ordered_indices, normalization
direction
expansion_variable, nf, color_factors, loop_order
source_check
```

`projector_or_definition` 记录论文的 continuum function；它不声明 caller 的
lattice raw vertex 使用了同一 projector。

## 3. MAPPED rows

### 3.1 Chetyrkin--Retey 2000

记录：arXiv:`hep-ph/9910332v2`，SU(3)，Landau gauge，
`a=alpha_s/(4*pi)=a_s/4`。

| Local symbol | Locator | Direction and algebra |
|---|---|---|
| `quark_field_conversion_ms_bar_over_rimom` | p. 11, Eq. (34) | literal `C_2^RI`; code terms `A2*a_s^2/16 + A3*a_s^3/64` |
| `quark_field_conversion_ms_bar_over_rimom_prime` | p. 11, Eq. (36) | literal `C_2^{RI'}`; no one-loop term and same `4^n` conversion |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | pp. 11--12, Eqs. (37), (41) | `m_MSbar=C_m^{RI'}m_RI'` |

Eq. (41) reference point: `nf=4`, `a_s=0.1`, decomposition
`1.0-0.1333-0.0701-0.0458=0.7508`.

### 3.2 Derived field ratio

`vector_conversion_ms_bar_over_rimom_prime` and
`quark_field_conversion_rimom_prime_over_rimom` are `MAPPED_DERIVED`:

```text
C_2^{RI'}/C_2^{RI} through O(a^3)
B2-A2 = -67/6 + 2*nf/3
B3-A3 = (-52321/18 + 607*zeta(3)
         + (8944/27 - 32*zeta(3))*nf
         - 208*nf^2/27)/4
```

Historical `vector_*` name does not bind a vector-current observable.

### 3.3 Gracey tensor

Gracey arXiv:`hep-ph/0304113v1`, p. 23, Eq. (4.11), `xi=0`:

```text
C_T = 1 + k2*a^2 + k3*a^3 + O(a^4)
C_T = Z_T^{RI'}/Z_T^MS
```

API returns the fixed-order inverse:

```text
C_T^{-1} = 1 - k2*a_s^2/16 - k3*a_s^3/64 + O(a_s^4)
```

The `/64` follows `a=a_s/4`; `/6` is not a valid variable conversion.

### 3.4 Coupling and MS mass

- BCK2016 arXiv:`1606.08659v2`, p. 1, Eqs. (1)--(5):
  `beta(a_s)=da_s/dln(mu^2)=-sum_i beta_i a_s^(i+2)`, `a_s=alpha_s/pi`.
- BCK2014 arXiv:`1402.6611v1`, pp. 3, 5, Eqs. (3.1)--(3.4), (4.7)--(4.12):
  MS mass anomalous dimension and c-function algebra.
- Returned orientation is
  `U=c(a_s(mu0))/c(a_s(mu))`.
- Scalar running is the local reciprocal wrapper, not an independent coefficient table.

## 4. Unverified legacy boundary

| Table group | Search record | Why still fail-closed |
|---|---|---|
| `quark_mass_conversion_*rismom*`, scalar reciprocals | Gorbahn--Jaeger arXiv:`1004.3997v2`, pp. 2--5, Eqs. (29)--(32) | local decimal path lacks exact RI/SMOM vs gamma_mu choice, `xi`, scale and special-function reduction |
| `tensor_conversion_*rismom*` | Gracey arXiv:`1101.5266v1`, pp. 16--18, Eqs. (6.1)--(6.8) | `RI'/SMOM` and tensor projector basis are not interchangeable with local `rismom` name |
| `*_prime2` variants | CR2000 Appendix B pp. 26--30; Gracey Eq. (4.11) candidates | name does not bind RI family, `xi` sign, direction or truncation |
| field/tensor full running outputs | CR2000 Eq. (7), Eqs. (50)--(52); Gracey Eq. (4.7) subsets | code factor-two/direction and zero padding/truncation are unbound |
| `matching_systematic_error` | no primary equation for seven slots/error transforms | no ordered basis or covariance prescription |
| Padé helpers | no derivation/order/domain/truncation found | literal numerical replay only |

CR2000 p. 4 Eq. (7), p. 16 Eqs. (50)--(52) map only the quark-field coefficient
subset `r0..r3` before a local factor of two. Gracey p. 21 Eq. (4.7) maps only
the MS tensor subset `r0..r2`. These rows are
`PRIMARY_COEFFICIENT_SUBSET_NONPROMOTING`, not physical profiles.

## 5. Promotion gate

A physical profile requires all of the following:

1. frozen paper version, page and equation;
2. ordered operator/projector basis;
3. source and target scheme, gauge, `nf`, color and normalization;
4. exact direction and reciprocal convention;
5. expansion variable and loop/truncation policy;
6. independent numerical reference point;
7. reviewer approval;
8. separate raw, conversion, running and statistical covariance artifacts.

Until then no legacy decimal table may be called verified physics.
