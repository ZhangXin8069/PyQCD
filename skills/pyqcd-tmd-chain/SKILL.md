---
name: pyqcd-tmd-chain
description: |
  Use when planning or auditing the end-to-end PyQCD gradient-flow nucleon-gluon
  TMD-PDF chain, from flowed gauge fields and OPE operators through disconnected
  ratios, Z_R or hybrid renormalization, Fourier/CS/matching, and continuum limits;
  use pyqcd-tmd-algorithm for implementation-level contracts and validation gates.
metadata:
  openclaw:
    emoji: 🌊
---

# pyqcd-tmd-chain — 梯度流胶子 TMD-PDF 全链导航

## 目的与边界

本技能是 PyQCD 核心物理链的路线图：说明每一阶段为何存在、消费什么、产出什么以及
何时可以进入下一阶段。它不复制算符几何、重整化公式或验证细节；实现契约统一放在
`pyqcd-tmd-algorithm` 及其 `references/`，统计执行放在 `pyqcd-statistics`。test9 持久
cache 的细节唯一见
[`pyqcd-tmd-algorithm/references/validation.md`](../pyqcd-tmd-algorithm/references/validation.md)。

目标量是梯度流方案下核子胶子 TMD-PDF：

\[
f_{g/N}^{[\Gamma]}(x,\boldsymbol b_\perp;\mu,\zeta)
\leftarrow h_g^{\rm flow}(z,\boldsymbol b_\perp;P_z,\tau,\ell).
\]

只有非零 `b_perp`、有限 staple 长度 `ell`、真实核子胶子三点、同几何 soft/rapidity
处理、胶子匹配和多尺度误差验证同时有证据，才可称为完整链；纵向直线结果应称为
quasi-PDF 或原型。

## 六阶段接口地图

| 阶段 | 代码入口 | 输入 → 输出 | 进入下一阶段的最小门 |
|---|---|---|---|
| 1. 流化 | `renorm/_gradient_flow.py` | `U` → `V_tau`、流能量 | 形状、幺正性、步长稳定性和流定义一致 |
| 2. 算符 | `operator/_gluon_ope.py`、`renorm/_tmd.py` | `V_tau` → Clover/对偶场强与 `O` | 端点、路径、Lorentz 指派、颜色闭合 |
| 3. 外态/断连 | `analysis/_tmd_ratio.py` | `C2`、`Lg`、`C2*Lg` → 真空扣除 ratio、`c0` | 共享组态索引、多个 `t_sep`、平台/协方差证据 |
| 4. 重整化 | `renorm/_zr.py`、`_hybrid.py` | `c0`/`hB`、soft → `Z_R`、`h_R` | 同流时间/表示/几何的方案和误差已写明 |
| 5. 提取/匹配 | `renorm/_tmdextract.py`、`_matching.py` | `h_R` → cos/sin quasi、CS、matched PDF | 两个 `Pz`、Fourier 偶奇、`α_s→0` 单位核 |
| 6. 连续极限 | `renorm/_extrapolate.py` | 多 `a/Pz/tau/ell` → 外推和误差带 | 协方差/系统窗/有限体积与拟合稳定性 |

令 `i,j` 为 `z_dir` 之外的两个空间方向，常用算符组合为

\[
O=M^{ti;ti}+M^{tj;tj}-2M^{ij;ij},
\]

其中 `z_dir=2` 时才简写为 `tx/ty/xy`。该组合不是完整张量混合矩阵的替代。
`gluon_ope_operator_z0`、固定规范 FF、螺旋度
算符、sin 型共线准 PDF 和 CS 两动量接口都是有明确边界的变体，不能互相冒充完整
TMD 结果。

## 推荐执行顺序

1. 先按 `pyqcd-conventions` 固定轴、单位、符号、边界和元数据，再读
   `pyqcd-tmd-algorithm` 的 geometry reference。
2. 各无量纲 `tau=t/a²` 都从同一 raw 组态独立生成 `V_tau`；核心算符 API 只做调用内
   复用。test9 若启用持久 cache，先读
   [`validation reference`](../pyqcd-tmd-algorithm/references/validation.md) 判定身份，再
   在同一 `V_tau` 上构造场强、路径和 Lorentz 分量，先保留复数，不提前取实部或强行偶化。
3. 外态阶段保留逐组态 `C2`、loop 和 `C2*loop`，在重采样后做断连真空扣除；统计细节
   转 `pyqcd-statistics`。
4. 通过几何/外态门后才应用 soft、`Z_R`/hybrid、Fourier、CS 和匹配；每一步保存中间量
   和尺度，不用默认 `soft=1` 代表已测量 soft。
5. 最后做多尺度外推并区分统计误差、拟合窗漂移、离散化、有限体积和截断系统误差；
   由 `pyqcd-docs` 记录证据和未验证项。

## 现有示例与验收入口

```bash
python examples/pyqcd/tmd_gradient_flow_demo.py
python examples/pyqcd/test9_gluon_tmd_nucleon.py --smoke
python examples/pyqcd/test9_verify.py <run_dir>
```

smoke/demo 只验证链路或形状；`test9_verify.py` 的 A–E 通过也不能替代真实非零横向
几何、soft/rapidity 和多尺度系统扫描。运行编排转 `pyqcd-pipeline`，后端/文件转
`pyqcd-infra`；持久 cache 不在本入口定义规则。

## 状态与降级措辞

| 状态 | 适用措辞 |
|---|---|
| 只有函数/文件 | “接口存在” |
| 受控输入断言通过 | “测试通过”或“原型” |
| 方案、尺度和 soft/rapidity 闭合 | “方案内结果” |
| 真实三点、多尺度和误差账本齐全 | “真实数据验证” |

任何阶段失败都保留 raw/intermediate 并停止升级状态；不要用文件齐全、图可生成或单一
系综的数值代替物理验证。

## 路由

几何/算法/物理门 → `pyqcd-tmd-algorithm`；规范流和涂抹 → `pyqcd-gauge`；核子算符与
缩并 → `pyqcd-physics-correlator`；统计 → `pyqcd-statistics`；批量运行 →
`pyqcd-pipeline`；结果报告 → `pyqcd-docs`。
