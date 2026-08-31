---
name: pyqcd-docs
description: |
  Use when writing or validating a Chinese PyQCD LaTeX note, analy/pure report,
  experiment report, or PDF deliverable, especially when xelatex, font coverage,
  source traceability, page rendering, Overfull, Float too large, or Missing
  character checks matter.
metadata:
  openclaw:
    emoji: 📄
---

# pyqcd-docs — 中文文档与报告

## 目的与边界

本技能负责把已验证的代码、数据和物理推理组织成可追溯的中文 LaTeX 文档，并完成
PDF 版式验收。它不生产规范场、传播子、统计结果或 TMD 数据；素材分别来自
`pyqcd-analysis`、`pyqcd-tmd-chain`、`pyqcd-pipeline` 和 `pyqcd-infra`。

按需读取：

| 任务 | Reference |
|---|---|
| xelatex、字体、三零与逐页 PDF 验收 | [`references/latex-validation.md`](references/latex-validation.md) |
| analy/pure/实战报告章节和图表映照 | [`references/report-structure.md`](references/report-structure.md) |

## 先确定文档类型

| 类型 | 目标 | 必需证据 |
|---|---|---|
| 研究笔记 | 公式、假设、极限和参考源 | 推导来源与量纲检查 |
| `analy` | 全库/模块结构和执行链 | 文件、行号、命令和实际输出 |
| `pure` | 核心算法穷尽剖析 | 代码直引、公式映射和边界 |
| 实战报告 | 数据、图、日志和物理结论 | 组态数、参数、产物、验证日志 |

## 最小证据契约

每条重要结论都应能回到“源码路径:行号、命令输出或产物文件”。代码片段优先用
`\lstinputlisting` 直接引用，不能手抄后声称逐字一致；报告必须区分“接口存在、受控
测试通过、方案闭合、真实数据验证”四种状态。单一系综、smoke 图或缺少 soft/rapidity
处理时，不得写成完整 TMD 物理结论。

## 工作流程

1. **定范围**：选择文档类型、读对应 reference，列出章节、输入和不在范围内的内容。
2. **收证据**：核实路径和行号，保存命令、shape、数值、日志和图表来源；先写状态表。
3. **写 tex**：正文使用统一字体和字号，公式标明假设/量纲/极限，图表配物理解释，
   不把截图堆成结论。
4. **编译**：在目标 `docs/` 或报告目录运行 XeLaTeX 两遍；不要用 pdflatex 代替中文编译。
5. **验收**：检查 `Overfull=0`、`Float too large=0`、`Missing character=0`，再核对页数、
   全页渲染、安全区、裁切、遮挡和图表可读性；失败就回到 tex 修复并重编。若由
   `pyqcd/pipeline/_steps.py::step_report` 生成，还必须遵守
   [`report-structure.md`](references/report-structure.md) 的缺失平台语义，以及
   [`latex-validation.md`](references/latex-validation.md) 的真实编译返回码和 PDF 新鲜度契约。
6. **归档**：按命名约定保存 `.tex/.pdf` 和必要的验证日志，更新对应管理文档；不修改
   外来参考库的既有内容。

## 统一版式与命名

- 主仓库/外来库的分析产物分别落在 `docs/` 与相应 `refer/git-rep/<库>/docs/`；
  文件名为 `analy_<slug>_<YYYYMMDD>` 或 `pure_<slug>_<YYYYMMDD>`。
- 正文约 10.54pt，表格 `small`，代码 `footnotesize`；禁用 `tiny`/`scriptsize` 逃避溢出。
- 等宽字体需覆盖源码中的希腊字母和 Unicode 符号；中文环境使用 XeLaTeX/ctex。
- `\quad` 后接中文留空格；图题、表题和交叉引用保持一致。

## 常见错误

| 现象 | 处理 |
|---|---|
| Overfull hbox | 改断行、列宽或图表布局，不靠缩小字体硬塞 |
| Float too large | 拆图/表、调整浮动体或移至附录 |
| Missing character | 核查字体覆盖、源码编码和 listing 设置 |
| 页数虽对但内容被裁切 | 渲染每页检查安全区、遮挡和字体可读性 |
| 结论无源码证据 | 回到素材清单，补路径:行号和实际命令输出；不能用记忆填空 |

## 交接

交付 `.tex/.pdf`、编译日志、三零和逐页验收结果、来源清单、未验证限制及管理文档
登记。数据链问题回 `pyqcd-analysis` / `pyqcd-tmd-chain`，运行问题回
`pyqcd-pipeline`，共享后端和文件问题回 `pyqcd-infra`。
