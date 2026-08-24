---
name: pyqcd-docs
description: |
  PyQCD 文档与报告技能：中文 LaTeX 文档编译约定（xelatex 两遍、Overfull=0/
  Float too large=0/Missing character=0 三零验收）、字体规范、analy/pure 分析
  报告产物命名与落盘约定、报告结构范式（15 步工作流框架、三视角、因果总览、
  图表-物理三段式映照）。触发于："写中文文档"、"编译 tex"、"出分析报告"、
  "analy 报告"、"pure 报告"、"PDF 交验"、"Overfull"。
metadata:
  openclaw:
    emoji: 📄
---

# pyqcd-docs — 中文 LaTeX 文档与分析报告

## 目的与边界

统一 PyQCD 中文 LaTeX 文档与分析报告（analy/pure）的编写、编译与交验标准，
保证产物三零验收达标、字体规范一致、引用可溯源。只管文档规范，不管数据生产
（上游见 pyqcd-analysis/pyqcd-tmd-chain/pyqcd-pipeline）。

## 编译与验收（硬闸门）

```bash
cd docs && xelatex <文档>.tex && xelatex <文档>.tex   # 一律 xelatex（中文），两遍
```

验收三零（log 中逐一 grep 确认）：**Overfull=0、Float too large=0、
Missing character=0**。任一非零必须修复后重编，不得带病交付。

## 字体与排版规范（dev5→dev5_2 教训）

- 字号分级：正文 10.54pt；表格 `small`；代码 `footnotesize`；
  **禁用 `\tiny`/`\scriptsize`**（字体过小是首版被打回的主因）。
- 等宽字体用 DejaVu Sans Mono——须覆盖 docstring 希腊字母与 ⊏/⊥ 等符号
  （默认 tt 字族缺字会触发 Missing character）。
- 中文排版：`\quad` 后跟中文需空格；文件名统一中文（docs/ 52 篇先例）。

## analy/pure 报告产物约定

| 项 | 约定 |
|---|---|
| 命名 | `analy_<slug>_<YYYYMMDD>.pdf` / `pure_<slug>_<YYYYMMDD>.pdf`（slug 拼音/英文短词） |
| 落盘 | 主仓库产 docs/；外来参考库产 refer/git-rep/<库>/docs/（新建文件，不改库内既有内容；非独立 git 仓库注明） |
| 编译 | xelatex 两遍 + 三零验收 |
| 引用 | 结论附参考源（`文件:行号`，引用前核实）；代码片段用 `\lstinputlisting` 直引原文件（逐字节一致） |

## 报告结构范式

- **15 步工作全流程框架**（主题为具体工作时）：A 代码/B 物理：目的与准备→输入→
  输出→框架→关键细节→正确执行→实际执行→实际输出→结果分析（图表/日志）→
  综合分析→总结→评价→补充→参考源→附录。
- **三视角**：主体结构 / 各部分关系 / 项目思路。
- **因果总览**（dev5_2 定型）：设「全链因果总览」节给出前因→后果主链，
  各节点五步展开（A5.x 式）。
- **图表-物理三段式映照**：每图按「物理意义 / 对应结果 / 物理解析」三段呼应
  关键公式（如 Eq.Odef Eq.quasiTMD Eq.Zij Eq.TMDmatch）。
- 物理内容占比目标 >60%（纯截图堆砌不合格）；公式推导逐步标注依据 +
  极限/量纲校验；代码-物理对象映射表（无孤儿符号）。

## 工作流程

1. 明确文档类型（笔记 / analy 全库分析 / pure 核心剖析 / 套件实战报告）→
   选定结构与模板章节。
2. 收集证据：实测命令输出、图、JSON 汇总；引用一律核实路径行号。
3. 写 tex → xelatex 两遍 → 三零 grep 验收 → 不达标修复重编（循环至通过）。
4. 产物登记：AGENTS.md 相应小节补一行（最小改动、先读后写），不代提交。

## 错误处理

| 场景 | 处理 |
|---|---|
| Overfull hbox | 改断行/缩表格/换列宽；禁止靠缩小字体硬塞 |
| Float too large | 表图拆分或转 sidewaystable/附录 |
| Missing character | 换 DejaVu Sans Mono 等全覆盖字族 |
| 中文文件名编译异常 | 检查 xelatex（非 pdflatex）与 ctex 加载顺序 |

## 与其他技能配合

- 数据与图的来源 → `pyqcd-analysis` / `pyqcd-tmd-chain` / `pyqcd-pipeline`；
- 全库/核心剖析的素材调查对应 configure 侧 analy/pure 技能（/root/configure/skills）。
