# AGENTS.md — examples/huangcl/docs

黄 CL 多步分析流水线代码分析文档（LaTeX）。

| 文件 | 说明 |
|---|---|
| `huangcl_code_analysis.tex` / `.pdf` | huangcl 流水线代码分析（~606 KB） |
| `analy_huangcl_formula_20260813.tex` / `.pdf` | 公式结构分析（analy，2026-08-13） |
| `analy_huangcl_physics_20260813.tex` / `.pdf` | 物理结构分析（analy，2026-08-13） |
| `pure_huangcl_20260815.tex` / `.pdf` | 核心部分穷尽剖析（pure，2026-08-15；15 页） |

## 编译

```bash
xelatex -interaction=nonstopmode -halt-on-error huangcl_code_analysis.tex   # 两遍
```
