# 状态、产品与 PyQCD 覆盖

## 1. 参考产品链

权威参考工作流根目录在历史环境中为：

```text
/public/group/lqcd/donghx/Diagram_GluonPDF
```

当前拟合输入应只使用 corrected `ratio_v4`，估计量 id 为
`post_ratio_six_direction_vdv_average_v1`，axes
`(boot,pabs,delta_z,tsep,tau)`。旧 ratio v2 和 bare-fit v1 含 C2 momentum-axis
错误，只能追溯，不能用于物理。

默认 Sumratio-difference 参考实现：

```text
Fit_SKILL/Sumratio_Difference_AIC06
Fit_SKILL/artifacts/sumratio_difference_aic06_fallback_v2/<ensemble>
```

其他参考模型：

```text
Fit_SKILL/Bare_Matrix_Fit       # undifferenced Sumratio + Ratio comparison
Fit_SKILL/AIC_Fit               # historical AICc sensitivity, not default
```

## 2. 参考实现状态

| 项 | 状态 | 证据范围 |
|---|---|---|
| strict physical `Tmin` | implemented, tested | L24x72/L32x64/L32x96 注册值和差分开端 |
| Ratio fit | implemented, tested | paired cross-check |
| Sumratio slope | implemented, tested | alternate/established model |
| Sumratio-difference AIC | implemented, tested | 固定 cut、omission penalty、逐 replica |
| complete bootstrap envelope | implemented, audited | 每个 retained point 的 filled coverage |
| quality vs fallback metadata | implemented, audited | 产品字段与图标记 |
| shared configurations/indices | tested | 全部 observables 共享 en/decoded index |
| physical validation of bare matrix | not established | 裸量本身不能替代重整化/物理验证 |

2026-09-03 的 v2 审计记录：27 个 NPZ、27 个 completion record、9 对
six-panel 图；7,830,000 个 filled bootstrap 值全部 finite；4,787 个 accepted
central points、433 个 central fallback，其中 288 个是 L24x72 `|P|=6` 的
zero-dof single-difference。该数字是历史快照；输入或代码变化后必须重跑审计。

## 3. PyQCD 当前覆盖

已实现且有现有测试的通用层：

- `pyqcd.analysis._ratio2pt.run_ratio2pt`：OPE/2pt vacuum subtraction、ratio、
  逐 `z` 拟合和报告。
- `pyqcd.analysis._bare_matrix.run_bare_matrix`：三方向 ratio 平均和拟合。
- `pyqcd.analysis._ratio_fit.fit_ratio`：correlated ratio fit 与逐样本拟合。
- `pyqcd.analysis._ratio_fit.fit_constant_window`：constant window diagnostic。
- `pyqcd.analysis._fitter`：fit、covariance rank、SVD 和 status 传播。
- `pyqcd.analysis._disconnected.resample` / `cov_mat` / `sem`：统计基元。

这些通用层能承担输入整理、拟合和回归，但当前 PyQCD **没有**以下专用实现：

- strict physical `Tmin=floor(threshold/a)+1` 的 bare-matrix contract；
- forward Sumratio-difference 的 omission-AIC engine；
- 逐 replica accepted/AIC/min-error/zero-dof fallback envelope；
- 完整 `M_quality` versus `M_filled` schema 与产品审计器。

因此不能把通用 `_bare_matrix` 或 `_ratio_fit` 的成功运行写成上述外部产品已迁移。
若任务要求这些契约，先实现并补 synthetic recovery、fallback-order、rank failure、
shared-index 和 product audit，再升级状态。

## 4. 最小验证命令

PyQCD 通用回归：

```bash
python -m pyqcd.testing
bash data/from-logs/logs/test0_ratio/run-local.sh
bash data/from-logs/logs/test0_bare/run-local.sh
```

相关单模块入口：

```bash
python -B -m pyqcd.testing._fit_identifiability_contract
python -B -m pyqcd.testing._fit_status_propagation_contract
python -B -m pyqcd.testing._bootstrap_resampling_contract
python -B -m pyqcd.testing._statistics_edge_contract
```

如果参考工作流可用，可按来源文档运行：

```bash
python run.py audit-inputs --ensemble L32x96
python run.py fit-all --ensemble L32x96
python audit_products.py
python -m unittest discover -s tests -v
```

这些命令可能依赖外部数据和环境；不能在 PyQCD 仓库内默认它们存在。

## 5. 审计清单

- 所有 expected channel/flavor products 和 completion JSON 存在。
- NPZ hash 与 completion record 一致。
- 每个 accepted fit 的 window 从固定 physical `Tmin` 开始。
- 没有 difference 读取更低 separation。
- `M_boot_filled` 在每个 retained point 有完整 finite replica。
- invalid accepted output 保持 NaN。
- 配置和 bootstrap index 在全部 observables 间一致。
- AIC 权重在相同固定 cut family 内归一化。
- central/replica fallback 和 zero-dof diagnostic 计数可复现。
- 图明确区分 accepted、single-cut 和 fallback。

## 6. 状态边界

外部的完整 v2 产品可在“声明模型与 artifact coverage”范围内标为 `closed`，但该
状态不等于裸矩阵元已经 `physical-validated`。PyQCD 的现有通用拟合层也只能标为
`implemented/tested`；专用契约尚未迁移。
