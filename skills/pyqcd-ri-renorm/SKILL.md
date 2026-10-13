---
name: pyqcd-ri-renorm
description: |
  Use when reviewing or implementing PyQCD RI-prime/MOM or nonexceptional
  RI/SMOM kinematics, independent two-leg amputation, projector-family
  selection, or raw quark-bilinear renormalization factors.
---

# pyqcd-ri-renorm — RI 投影与 raw factors

## 目的与边界

本技能负责强子无关的 quark-bilinear NPR 前置层：固定动量学、独立外腿截肢、投影
family、tree normalization，以及 `Z_q/ZA/ZV/ZS/ZP/ZT` 的 raw factors。

- 输入是已完成规范固定、动量源求解和 Fourier 转换的 propagator/Green function。
- 输出是保留组态或重采样轴、完整 provenance 和 scheme identity 的 raw 对象。
- 本技能不负责 gauge fixing、传播子求解、MPI reduction、window 拟合、协方差估计或
  `MS-bar` conversion/running。
- 当前源码证据只支持树图代数和输入域验证；不能据此宣称 interacting NPR 已完成。

## 适用与不适用

| 适用 | 不适用或必须转交 |
|---|---|
| 验证 `p_in/p_out/q/mu`、例外或非例外动量学 | 构造动量源、求逆或 MPI 汇聚 |
| 检查两条独立外腿的 amputation | 用一条 propagator 伪造一般非例外第二腿 |
| 明确 `gamma_mu` 与 `qslash` projector family | 仅凭 projector 名或 `omega` 宣称标准 scheme |
| 计算 tree-normalized raw factors | 输出最终 `MS-bar` 或 TMD 物理结果 |
| 定义 `unmapped_fail_closed` 状态 | 跳过 provenance 自动外接匹配核 |

## 核心公式与归一

采用 Euclidean 约定

$$
q=p_{\rm out}-p_{\rm in},\qquad
p_{\rm in}^2=p_{\rm out}^2=\mu^2>0,\qquad
q^2=\omega\mu^2,\quad \omega\ge 0 .
$$

`omega=0` 是例外运动学并只使用 `gamma_mu` family；`omega=1` 是对称非例外点；
其他 `omega>0` 是推广非例外点。必须同时记录格点动量定义，例如
`ap=2*pi*n/L`、`sin(ap)` 或改进动量，不能只保存整数模式。

末四轴固定为

$$
(\text{sink spin},\text{source spin},
  \text{sink color},\text{source color})=(4,4,3,3),
$$

并映射为 compound `(s,c)` 的 `12 x 12` 矩阵。独立双腿截肢为

$$
\Lambda_O(p_{\rm out},p_{\rm in})
=S^{-1}(p_{\rm out})\,G_O(p_{\rm out},p_{\rm in})\,S^{-1}(p_{\rm in}).
$$

`S_in` 与 `S_out` 必须是独立测量输入；不得用 `gamma5` Hermiticity 从一个传播子
制造一般非例外第二条腿。

单通道 tree 归一与 propagator-based `Z_q` 为

$$
\lambda_O=
\frac{\operatorname{Tr}(P_O^\dagger\Lambda_O)}
     {\operatorname{Tr}(P_O^\dagger\Lambda_O^{(0)})},
\qquad
Z_q^{\rm prop}(p)=
-\frac{i}{12p^2}\operatorname{Tr}_{s,c}
\left[(\slashed p\otimes I_c)S^{-1}(p)\right].
$$

raw factors 为

$$
Z_O=\frac{Z_q}{\lambda_O},\qquad
O\in\{A,V,S,P,T\}.
$$

`gamma_mu` family 使用 `gamma_mu` 与 `gamma_mu*gamma5`。`qslash` family 使用

$$
P_{V,\mu}=q_\mu\slashed q,\qquad
P_{A,\mu}=q_\mu\slashed q\,\gamma_5,
$$

其 `gamma5` 顺序必须与 tree vertex 逐项核对。`Zq` 可选择 incoming、outgoing、
arithmetic mean 或 principal-branch geometric mean；后两者是局部统计约定，不是
标准 scheme 定义。复数几何平均不得静默取实部、绝对值或切换 branch。

## 工作流

1. 冻结 gauge fixing functional、残差、残余规范处理、Gribov-copy policy、gamma
   basis、边界条件、Fourier 符号、源汇 ownership 与动量定义。
2. 独立生成 `S(p_in)`、`S(p_out)` 和 `G_O(p_out,p_in)`，保存 solver residual、
   iterations、失败求解与组态 ID。
3. 建立不可变的 kinematics record，检查 `q`、`mu^2`、`omega` 和 projector family；
   手工构造 record 也必须在下游防御性复验。
4. 将末四轴映射到 `12 x 12`，分别求逆并计算
   `S_out^{-1} G_O S_in^{-1}`。禁止隐式 batch broadcasting。
5. 按明确公式投影 S/P/V/A/T；V/A 四分量、T 六个 `mu<nu` 分量及顺序必须冻结。
6. 选择具名 `Zq` convention，计算 raw factors，并写入完整 `scheme_identity`。
7. 保留逐组态或逐重采样结果，连同动量、投影器和 tree kernels 一并交给统计层。

## 验收门

- [ ] `q=p_out-p_in`，三条 invariant 均通过声明容差。
- [ ] 两条外腿来自独立输入，交换或复用一腿的 mutant 能被检测。
- [ ] spin-color 轴顺序、`12 x 12` compound mapping 和 batch shape 精确一致。
- [ ] projector、tree vertex、gamma basis 和 Dirac 矩阵均为有限数值，且同后端。
- [ ] free-field/tree-level 的全部 raw channel 回到 1；只把此作为代数 oracle。
- [ ] `qslash` axial projector 的 `gamma5` order 与目标公式一致。
- [ ] tensor 六分量顺序和 normalization 不凭数组位置猜测。
- [ ] `scheme_identity` 明确写出 kinematic label、projector family、`Zq` 规则及
  custom/unmatched 状态。
- [ ] `continuum_matching_status` 为 `unmapped_fail_closed`；没有 exact equation map
  时不得替代为标准 RI/SMOM matching。
- [ ] 任何 interacting、gauge-fixed、continuum 或 `MS-bar` 结论都保持“未验证”。

## 与 PyQCD 技能交接

| 技能 | 交接责任 |
|---|---|
| `pyqcd-gauge` | 提供规范链接、规范固定状态、路径/方向/边界约定；缺少可靠 gauge provenance 时停止 raw NPR 升级。 |
| `pyqcd-statistics` | 接收逐组态或逐 resample 的 amputated vertices/raw factors，执行共享索引的 jackknife/bootstrap、协方差和秩门。 |
| `pyqcd-analysis` | 只消费已带状态字段的中心值、误差、窗口和不可辨识标记；图表不得把 raw factor 重标为最终物理量。 |
| `pyqcd-tmd-algorithm` | 仅在 exact conversion/running profile 已建立后接入 TMD 链；不得把 custom/unmatched RI 因子直接拼入软因子、CS 核或胶子匹配。 |

交给下游的 manifest 至少包含：输入 hash、gauge/动量/source 约定、独立腿身份、
projector/tree matrices、`Zq` convention、scheme identity、样本轴、协方差和状态。

## 来源与证据上限

来源技能冻结于 `LQCD_Skills` 提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。详细公式、实现边界与逐函数证据见
[`references/physics-contract.md`](references/physics-contract.md)。

主来源为 arXiv:0901.2599 的 RI/SMOM scheme 定义，以及
arXiv:2310.00814v2 Appendix B.1, pp. 13--14, Eqs. (32), (34) 的
propagator-`Zq` 与 independent-leg amputation。后文不背书本技能的 custom
`qslash`、腿平均或未映射 projector 组合。

当前证据上限：E1，树图 CPU 代数、形状和输入域。尚无 interacting gauge-fixed
数据、真实 GPU/MPI/QUDA solve、连续极限或匹配闭环；因此不能声称生产级 RI/SMOM
或最终 `MS-bar` 结果。
