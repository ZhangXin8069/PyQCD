---
name: pyqcd-quark-renorm
description: |
  Use when auditing or staging PyQCD quark-bilinear RI-to-MSbar conversion,
  perturbative running, profile-registry provenance, or legacy quark
  renormalization coefficient tables.
---

# pyqcd-quark-renorm — conversion、running 与 profile registry

## 目的与边界

本技能只处理 raw RI factor 之后的 continuum 后处理：

```text
raw gauge-fixed RI factor
  -> chiral/orbit/a2p2 treatment with covariance
  -> exact conversion at mu
  -> exact running from mu to target scale
```

- raw amputation、projector、`Zq` 和 tree normalization 属于 `pyqcd-ri-renorm`。
- 本技能接受已核实的 raw samples/covariance，输出可追溯的 converted/running factor。
- 未映射的历史 decimal tables 只能作为 literal/audit replay，不能称为已验证物理。

## 适用与不适用

| 适用 | 不适用或必须转交 |
|---|---|
| 建立或审查 coefficient profile registry | 构造传播子、做独立双腿 amputation |
| 核对 paper version/page/equation/direction | 仅凭函数名或 decimal 相似度决定 scheme |
| 计算已映射的 conversion 与 running | 将 conversion、running 和统计误差合成一个无标签数 |
| 执行公式级 numerical checkpoint | 为未映射 Padé/systematic table 提供 production number |
| 保存分阶段 provenance | 用 `physical=True` 冒充 interacting NPR 已验证 |

## 核心公式与归一

全局耦合 expansion 变量必须逐函数写明：

```text
a_s = alpha_s/pi
a   = alpha_s/(4*pi) = a_s/4
```

若 conversion 定义为

$$
C_O^{\rm target\leftarrow source}(\mu)
=\frac{Z_O^{\rm target}(\mu)}{Z_O^{\rm source}(\mu)},
$$

则只有 direction 已核对后才能使用 `Z_target=C_O Z_source`。函数名中的
`ms_bar_over_*` 只是历史 API 方向线索，不能替代原始方程和 source/target 语义。

运行因子采用

$$
U(\mu,\mu_0)=\frac{c(a_s(\mu_0))}{c(a_s(\mu))},
$$

并保持标度方向、`nf`、`Lambda`、loop order 和 truncation。逐截断求和一般只满足
同一阶的近似 composition；不能要求独立截断级数精确可乘。

## Profile registry

每个可进入物理阶段的 conversion/running API 必须先登记一行 profile。最低字段为：

```text
symbol, status
paper, version, page, equation
source scheme, target scheme, gauge
operator/definition basis and ordered indices
direction and normalization
expansion variable, nf/color content, loop order
independent numerical checkpoint
```

状态只允许：

| 状态 | 含义 |
|---|---|
| `MAPPED` | 论文版本、页码、方程、变量和代码代数逐项核对 |
| `MAPPED_DERIVED` | 由已映射 rows 的明确代数推导，推导步骤可复现 |
| `UNVERIFIED_LEGACY` | 数字或分支没有完整对象、方向、scheme/equation 依据 |

`physical=True` 只能表示“formula provenance 已登记”，不表示 lattice projector、
interacting NPR、协方差或连续极限已完成。

## 已映射边界

| API/profile | 状态 | version/page/equation | direction |
|---|---|---|---|
| `quark_field_conversion_ms_bar_over_rimom` | `MAPPED` | CR2000 v2, p. 11, Eq. (34) | literal `C_2^RI` polynomial；外部 `Zq` direction 仍需 caller binding |
| `quark_field_conversion_ms_bar_over_rimom_prime` | `MAPPED` | CR2000 v2, p. 11, Eq. (36) | literal `C_2^{RI'}` polynomial |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | `MAPPED` | CR2000 v2, pp. 11--12, Eqs. (37), (41) | `m_MSbar=C_m^{RI'} m_RI'` |
| `vector_conversion_ms_bar_over_rimom_prime`; `quark_field_conversion_rimom_prime_over_rimom` | `MAPPED_DERIVED` | CR2000 v2, Eq. (36)/Eq. (34) | fixed-order `C_2^{RI'}/C_2^{RI}`；名称不绑定 vector-current observable |
| `tensor_conversion_ms_bar_over_rimom_prime` | `MAPPED` | Gracey v1, p. 23, Eq. (4.11) | source `C_T=Z_T^{RI'}/Z_T^MS`；API 返回 fixed-order inverse |
| five-loop beta / coupling | `MAPPED`; `MAPPED_DERIVED` | BCK2016 v2, p. 1, Eqs. (1)--(5) | continuum `a_s=alpha_s/pi` |
| MS mass / scalar running | `MAPPED`; `MAPPED_DERIVED` | BCK2014 v1, pp. 3, 5, Eqs. (3.1)--(3.4), (4.7)--(4.12) | MS mass c-function 与 scalar reciprocal |

所有 `rismom`/`rismom_mu`、`*_prime2`、scalar/tensor Padé、
`matching_systematic_error`，以及未绑定对象的 field/tensor full running output，
均保持 `UNVERIFIED_LEGACY`。公式相邻或 decimal 相同不构成 mapping。

## 工作流

1. 从 `pyqcd-ri-renorm` 接收 raw samples、完整 scheme identity、momenta、projector
   与 covariance manifest；缺失任一项时停止 conversion。
2. 选择目标 profile，先核对 status、paper version、page、equation、operator basis
   和方向；`UNVERIFIED_LEGACY` 只能做 literal/audit replay。
3. 将 expansion variable、`nf`、gauge、loop order、`Lambda` 和 scale direction
   写入 manifest，并用独立数值参考点复现。
4. 在固定 `mu` 执行 conversion，单独保存结果；不要同时运行到最终标度。
5. 在明确 `mu -> mu0` 方向执行 running，分别保存 coupling、conversion 和 running。
6. 将逐样本结果交 `pyqcd-statistics`；统计、拟合窗口、截断、标度与 `Lambda`
   系统误差分别报告。
7. 只有全部 profile 字段、参考点和分阶段产物齐全时，才把结果交给 TMD 链。

## 验收门

- [ ] registry 中不存在待猜的 version、page、equation 或 direction。
- [ ] `a_s` 与 `alpha_s/(4*pi)` 未混用；每项分母可由方程逐项重建。
- [ ] reference checkpoint 独立复算，不只比较相邻函数。
- [ ] conversion 与该 conversion 之后的 running 分阶段保存。
- [ ] `UNVERIFIED_LEGACY` 分支保持显式拒绝或 literal 标签。
- [ ] scalar reciprocal 只在同一 mass/scalar scheme、gauge、flavor 和 truncation 下使用。
- [ ] tensor API 的 inverse direction 与 source convention 明确。
- [ ] 协方差与样本索引来自 `pyqcd-statistics`，没有把 diagonal errors 冒充 full covariance。
- [ ] 结果不得声称“legacy decimal tables 已验证”。

## 与 PyQCD 技能交接

| 技能 | 交接责任 |
|---|---|
| `pyqcd-gauge` | 提供规范作用量、gauge fixing、链接方向/边界与 ensemble provenance；fermion action 不能决定 gauge normalization。 |
| `pyqcd-tmd-algorithm` | 接收已映射的 conversion/running profile 与阶段化产物；raw RI 或未映射系数不得直接进入 TMD matching。 |
| `pyqcd-statistics` | 对 conversion/running 前的逐样本输入执行共享索引重采样、完整协方差、SVD/秩门和系统误差分解。 |
| `pyqcd-analysis` | 生成拟合和图表时必须保留 profile status、窗口、协方差和未覆盖项；不得把 literal replay 画成 production result。 |

## 来源与证据上限

来源技能冻结于
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。逐行映射、未映射表和失败条件见
[`references/profile-and-source-ledger.md`](references/profile-and-source-ledger.md)。

当前证据上限：

- spin-color array algebra 与选中 series 为 E1；
- `MAPPED/MAPPED_DERIVED` 是公式 provenance，不是 interacting NPR；
- 其余 decimal/Padé/systematic 表为 E0 literal contract；
- 尚缺 gauge-fixed interacting data、连续外推、阈值匹配和完整误差预算。
