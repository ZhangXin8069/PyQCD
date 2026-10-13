---
name: pyqcd-bare-matrix
description: Use when a PyQCD task selects a physical Tmin, builds Ratio, Sumratio, or Sumratio-difference fits, assigns AIC weights, completes bootstrap coverage, separates accepted fits from fallback, or audits shared configurations and bootstrap indices.
---

# pyqcd-bare-matrix - 裸矩阵元拟合契约

## 目的与边界

本技能把已经声明好算符、投影、方向和统计估计量的 ratio 产品变成未重整化矩阵元
及其质量/fallback 诊断。输出仍是 bare matrix element，不是重整化矩阵元、准 PDF
或物理 PDF。

输入必须携带组态清单、bootstrap index、估计量 id、mask、轴、算符约定和上游版本。
如果这些身份不完整，先在 `pyqcd-statistics` 和 `pyqcd-conventions` 中补齐，不进入
本技能的拟合链。

状态词只按下列含义使用：

- `implemented`：接口、脚本或产品 schema 存在。
- `tested`：有合成数据、单元、审计或回归证据，限定到实际检查范围。
- `closed`：声明模型、窗口、质量门、fallback、输出字段和审计链已完整覆盖。
- `physical-validated`：真实系综的 bare matrix 结果经过独立物理/系统检查；仅拟合
  成功或 bootstrap 全覆盖不算。

## 按需参考

- 读 [references/fit-contract.md](references/fit-contract.md) 获取
  Ratio/Sumratio/Sumratio-difference、AIC、complete bootstrap 和 fallback 的完整定义。
- 读 [references/status-and-products.md](references/status-and-products.md) 获取
  外部参考产品、PyQCD 当前覆盖、审计字段和验证命令。

## 最小工作流

1. **审计输入**：验证估计量 id、ratio axes、valid mask、组态顺序、bootstrap index、
   投影、方向平均阶段、单复数投影、算符公式和整体符号。
2. **计算物理 `Tmin`**：固定 `tmin_fm_threshold`，由 lattice spacing 计算
   `Tmin=floor(threshold/a_fm)+1` 并验证 `Tmin*a_fm > threshold`。所有候选窗口从
   同一 `Tmin` 开始，只扫描上界。
3. **全局选模型**：production 只选一个主模型。可注册 Ratio、Sumratio 或
   Sumratio-difference；不能逐点按误差、Q 或外观选方法。
4. **构造配对样本**：保持原 configuration bootstrap，所有方向、`z`、`p`、偶奇、
   flavor 和方法使用同一个 `(Nboot,Nconf)` index 数组。
5. **估计协方差**：使用声明的 covariance estimator（默认 Schaefer-Strimmer）
   和 SVD cut；保存 raw rank、effective rank、condition number、shrinkage、
   eigen spectrum、`chi2`、dof、Q。
6. **扫描候选窗**：在同一 cut 内比较嵌套 diff range，使用 omission-penalized AIC；
   不同 cut 改变数据向量和 likelihood normalization，不跨 cut 直接 AIC。
7. **逐 replica 选择**：每个 `(model,channel,momentum,z,replica)` 都先尝试质量门；
   没有 accepted candidate 时按固定 fallback 规则填完整数组，并保持 fallback 标记。
8. **交叉检查**：若比较 Ratio 与 Sumratio，使用 paired bootstrap difference/pull，
   不把两个独立误差相加，也不挑更漂亮的方法。
9. **落盘与报告**：同时保存 accepted、filled、quality、fallback、window/cut、
   covariance、输入 hash、代码 hash、命令和未决状态。

## 验收门

1. 输入估计量必须是当前版本。带已知 momentum-axis 错误的历史产品只能作追溯。
2. `ratio_valid_mask` 和 NaN 位置必须保留；无效点不能填零后参与 fit。
3. `Tmin` 必须由物理阈值计算并与结果一起保存；不能由每个点后验选择。
4. 候选必须有有效、有限的 mask 子集，协方差/设计可辨识，正自由度，满足注册的
   `Q`、`chi2/dof` 和 bootstrap success 门。
5. 每个保留点必须有完整 replica 结果；有 fallback 时保存 fallback candidate、
   原因和 fraction，不能把 filled 结果写成 quality-accepted。
6. AIC 权重只在相同 cut、相同 observable/likelihood family 内归一化；逐 replica
   重算，不能复用 central-data 权重。
7. accepted error 为 `std(samples,ddof=1)`，不再除 `sqrt(Nboot)`。
8. Ratio/Sumratio 方法比较保持相同配置、index、cut、窗口、方向平均和可辨识门；
   paired pull 超阈值时报告 unresolved dependence。
9. 图中的 accepted、single-cut、fallback 和 NaN 必须视觉区分。

## 当前实现边界

| 能力 | 参考工作流 | PyQCD 当前包 |
|---|---|---|
| generic ratio / correlated fit | implemented, tested | `pyqcd.analysis._ratio2pt`、`_bare_matrix`、`_fitter` 已实现并有回归 |
| shared bootstrap index 纪律 | implemented, tested | `pyqcd-statistics` 和 resample 契约已实现 |
| strict physical `Tmin` gate | implemented, tested | 没有此专用契约 |
| Sumratio-difference omission AIC | implemented, tested, v2 product closed | 没有专用实现 |
| complete per-replica quality/fallback envelope | implemented, audited | 没有专用实现 |
| independent physical validation of extracted bare matrix | no | no |

详细状态和产品字段见
[references/status-and-products.md](references/status-and-products.md)。

## 交接

- `pyqcd-analysis`：负责载入 ratio、运行图/表和分析输出；本技能定义拟合模型和选择门。
- `pyqcd-statistics`：负责 resampling、协方差、SVD、rank/identifiability 和通用
  fit status；本技能规定裸矩阵元专用 AIC/fallback。
- `pyqcd-physics-correlator`：定义 C3/C2、投影、方向平均和算符约定。
- `pyqcd-propagator`：只在输入数据缺失、需要重算 C2/C3 或顺序源时介入；拟合层不
  重新求解传播子。
- `pyqcd-infra`：负责 HDF5/二进制读写、dtype、MPI 归约、batch 和持久化审计。
- 上游梯度流夸克链：读 `pyqcd-quark-pdf`；拟合只消费其声明过的 `ratio_v4`。

## 来源清单

- `skills/donghx/fit_bare/SKILL.md`
- `skills/donghx/fit_bare/references/default-difference-aic.md`
- `skills/donghx/fit_bare/references/local-products.md`
- `skills/donghx/fit_bare/references/alternative-models.md`
- `skills/donghx/quark-pdf-bare-matrix/references/fit-selection.md`
- `skills/donghx/quark-pdf-bare-matrix/references/local-pipeline.md`
- `skills/donghx/quark-pdf-bare-matrix/references/physics-conventions.md`
- `skills/donghx/lqcddb/references/statistics-and-spectroscopy.md`
- `skills/sush/lqcddb/references/statistics-and-spectroscopy.md`
- `skills/donghx/Markdown/gradient_flow_gluon.md` 第 11 节
