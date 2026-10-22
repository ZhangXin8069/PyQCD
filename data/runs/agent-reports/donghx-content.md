# Donghx 其他内容只读审计

审计日期：2026-10-22（Asia/Shanghai）
范围：`refer/donghx/**` 中除主 PDF 提取链外的源码、Note、`_codex` 和
历史文档；`refer/git-rep/**` 未进入审计范围。

## 结论

`refer/donghx` 是只读历史快照，不是有固定上游 commit 或许可证的发布源。
`copy.sh` 只记录了集群绝对路径；本地 Git 能证明的是归档提交 `fc3a7c9`
同时删除了旧顶层 `Operator.py`、`2pt_proton_*.py` 等文件，不能证明
当前目录对应 Donghx 集群的某个固定版本。

## 来源与处置

| 内容 | 处置 |
|---|---|
| `_Gradient_Flow_GluonPDF` | `reference-only`；吸收物理约定、统计契约和失败状态，不复制为 canonical 源码。 |
| `Note/Gradient_Flow`、`Note/Quark_PDF` | 作为物理约定和公式参考；与当前 PyQCD 代码冲突处必须显式注册。 |
| `_codex/Markdown` | 日期化 AI/Codex 交接记录；只吸收窗口、统计、状态和 line-mass 边界。 |
| `_codex/skills` | 吸收审计契约；不原样安装为 PyQCD 权威技能。 |
| `docs/*.tex/.pdf` | 历史静态阅读或派生报告；不能作为当前数值验证。 |
| `AGENTS.md`、`docs/AGENTS.md` | 索引仍引用已删除的旧顶层源码，保留但不得据此重编译或运行。 |

目录内未发现 LICENSE/COPYING；`.system` 下部分系统技能有独立 Apache-2.0
许可，但不覆盖 Donghx 快照。

## 会误导最终物理结论的条目

- “完整、自洽、可交付胶子 PDF 链”的旧结论没有当前可重放的数值、协方差、
  soft/rapidity、mixing 或连续极限证据。
- 六种 Lorentz/方向算符变体不等于 TMD 快度演化、软函数、CS 核和 TMD
  matching 已实现。
- 231 conf、406 conf、1700 conf 等分支使用不同流时、统计量和物理目标，
  不能横向平均或互相替代误差。
- `RESULTS.md` 的 `P/F` 是拟合质量状态，不是物理验收；`VALIDATION.md`
  的“全部通过”主要验证索引、轴、hash 和文件完整性。
- `NO_Z0_NORMALIZATION.md` 已明确禁止把 `M(0)` 当作 hybrid 的 Pz=0 分母。

## 可吸收内容

- `finite-flow bare C3/C2`、GF-to-MSbar quasi/pseudo 矩阵元、light-cone PDF
  三层分离，不得跨层命名。
- `quality_accepted`、`selected_available`、`fallback_used` 分离存储。
- 完整组态、共享 bootstrap 索引、`ddof=1`、不除 `sqrt(Nboot)`、不删除失败
  replica。
- `c3_ratio_v3`、`ratio_v4`、`bare_matrix_fit_v3` 等版本化产品契约；旧
  ratio v1/v2 只作追溯。
- 线质量、`Z_chi`、端点系数和 LaMET matching 是不同层次；`C3/C2` 不自动
  消除 Wilson-line 线质量。
- 流时间窗口、`lambda` 尾部模型和连续极限必须作为全局系统误差变化，不能
  逐点挑选最小误差窗口。

## 当前边界

- 多数 `/public` 路径和符号链接在当前环境失效；旧结果 NPZ/CSV/manifest
  多数缺失。
- 上游版本、许可证和原始工作区来源未闭合。
- 本轮只把这些内容用于代码/公式审计；任何最终物理量仍必须由当前 PyQCD
  链重新生成并通过严格门。
