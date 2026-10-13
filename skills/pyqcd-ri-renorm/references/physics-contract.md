# RI 物理契约与证据台账

## 1. 状态

- 来源提交：`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。
- 状态：`CUSTOM_SCHEME_FAIL_CLOSED`。
- 最高直接证据：E1，树图 CPU 代数、shape/backend 与拒绝路径。
- 未验证：interacting gauge-fixed NPR、GPU/MPI、连续极限和
  `MS-bar` conversion。

## 2. 动量学和 identity

定义 `q=p_out-p_in`，要求

```text
p_in^2 = p_out^2 = mu^2 > 0
q^2 = omega * mu^2, omega >= 0
```

`omega=0` 标为例外/`RIprime/MOM` 并限定 `gamma_mu`；`omega=1` 标为对称非例外；
其他正值标为推广非例外。`momentum_definition` 必须显式记录：

```text
ap_mu = 2*pi*n_mu/L_mu
p_hat = sin(ap)
p_tilde = 2*sin(ap/2)
```

这些标签不是完整 scheme identity。完整 identity 还必须包含 gauge、gamma basis、
incoming/outgoing sign、source construction、`Zq` definition、projector 与
normalization equation。

## 3. 双腿代数

末尾布局：

```text
(..., sink_spin, source_spin, sink_color, source_color)
= (..., 4, 4, 3, 3)
```

compound index：

```text
row    = (sink_spin, sink_color)
column = (source_spin, source_color)
```

amputation：

```text
Lambda_O = S_out^{-1} G_O S_in^{-1}
```

独立腿要求精确 shape/backend equality，禁止用 `(1,...)` 隐式 broadcast。一般
非例外 Green function 的第二条腿必须来自 production workflow；不能由
`gamma5 S^dagger gamma5` 或单条 propagator 替代。

## 4. Projector 与 raw factor

通用 tree normalization：

```text
lambda_O = Tr(P_O^dagger Lambda_O)
           / Tr(P_O^dagger Lambda_O^tree)
raw Z_O   = Zq / lambda_O
```

多分量 multiplet 先累加 numerator 与 denominator，再做一次比值；不得先归一化
各分量后简单平均。

`gamma_mu` family：

```text
tree/projector V_mu = gamma_mu
tree/projector A_mu = gamma_mu gamma5
```

`qslash` family 的局部 kernel：

```text
P_V,mu = q_mu qslash
P_A,mu = q_mu qslash gamma5
```

tree vertex 仍使用 `gamma_mu` 和 `gamma_mu gamma5`。纸张对
`gamma5` order、Euclidean factor 或 Ward identity 的定义不同时，必须先逐式
映射，不能按名称替换。

`Zq` 的局部选择：

```text
incoming
outgoing
arithmetic_mean = (Zq_in + Zq_out)/2
geometric_mean  = principal_sqrt(Zq_in * Zq_out)
```

复数几何平均不得 silent `real`、`abs` 或 branch replacement。propagator-`Zq`
不是 vector-vertex `Zq`。

## 5. Scheme identity 与 fail-closed

局部结果的 identity 形如：

```text
<kinematic/projector family>
__Zq_propagator_<leg rule>
__custom_unmatched
continuum_matching_status = unmapped_fail_closed
```

`omega` 与 projector family 不足以生成标准 `RI/SMOM_gamma_mu` 或
`RI/SMOM_qslash` matching。必须逐式实现 vector-vertex `Zq` 或提供已核对的
continuum scheme map，之后才可改变状态。

## 6. 来源定位

| 内容 | 来源与页码/方程 | 支持范围 |
|---|---|---|
| nonexceptional RI/SMOM | arXiv:0901.2599, scheme definitions, Eqs. (1), (5), (9)--(12) | method class、scheme 条件；不验证所有局部 decimal/projector |
| propagator `Zq` | arXiv:2310.00814v2 Appendix B.1, p. 13, Eq. (32) | 给定 Fourier/propagator normalization 下的公式 |
| independent-leg amputation | arXiv:2310.00814v2 Appendix B.1, p. 14, Eq. (34) | amputation 结构 |
| 示例 projector factors | arXiv:2310.00814v2 Eqs. (35)--(42) | 仅论文指定 convention，不 blanket 支持 custom family |
| 轴序、形状和局部公式 | 来源技能脚本 `Def_ri_renorm.py` 的只读审计 | 实现定义，不是物理验证 |

## 7. 必须保留的反例与检查

1. swapped-leg mutant 必须与正确独立腿分离。
2. 非对角 color propagator 必须参与 dagger/compound-index 检查。
3. complex `q/mu^2`、Boolean provenance 和非有限输入必须在数值内核前失败。
4. tensor component 顺序与 V/A 四分量顺序必须由调用者冻结。
5. interacting `ZV=ZA`、`Zm ZP=1` 或 Ward identity 只有在 chiral limit、
   regularization symmetry、current normalization、gauge 和 scheme 全部匹配时才有意义。

来源所述的 tree-level tests 只证明局部代数；不能升级为 production NPR 证据。
