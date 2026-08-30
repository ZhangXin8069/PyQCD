# AGENTS.md — docs

格点 QCD 中文 LaTeX 笔记（51 篇，xelatex 编译；文件名统一为中文）。

## 文件

| 类别 | 文件（节选） |
|---|---|
| 胶子 PDF 总论 | `胶子PDF笔记.tex`、`胶子PDF完整推导.tex`、`胶子PDF计算代码解析.tex` |
| 连续极限 | `胶子PDF连续极限.tex`、`胶子PDF连续极限幻灯片.tex`、`胶子PDF幻灯片.tex`、`胶子PDF连续极限论文.tex`、`胶子TMD梯度流连续极限.tex` |
| TMD/重整化 | `格点QCD中的TMD_PDF.tex`、`格点QCD中的梯度流重整化.tex`、`格点QCD中的重整化.tex` |
| 理论框架 | `格点QCD中的光锥PDF、准PDF、赝PDF.tex`、`格点QCD中的光锥PDF与quasi_PDF.tex`、`格点QCD中的大动量有效理论.tex`、`格点QCD中的光锥与光前.tex` |
| 格点技术 | `格点QCD中的Wilson线.tex`、`格点QCD中的场强张量.tex`、`格点QCD中的胶子算符.tex`、`格点QCD中的smear算法.tex`、`格点QCD蒸馏方法解析.tex` |
| 方法学 | `格点QCD中的外推.tex`、`格点QCD中的误差统计.tex`、`格点QCD中的重采样方法.tex`、`格点QCD中的蒙卡方法.tex` |
| 仓库/工作流 | `PyQCD仓库结构与核心物理链解析.tex`、`理论解析与工作流.tex`、`构造胶子准算符.tex`、`格点上计算胶子准算符.tex` |
| 分析与专题 | `质子自旋危机解析.tex`、`格点QCD中的胶子极化.tex`、`格点QCD中的部分子分布函数.tex` |

完整清单：`ls docs/*.tex`（51 篇）；`*.aux/*.log/*.out/*.nav/*.snm/*.toc` 为编译产物（gitignore）。

## 编译

```bash
cd docs && xelatex -interaction=nonstopmode <文件>.tex   # 中文必须 xelatex；两遍
```

## 约定

- 文件名统一中文（英文名已迁移时中文化）；tex 内路径引用指向 `/root/PyQCD/`。
- 文本模式数学符号需 `$...$` 包裹（如 `$\pi$介子`）；`\quad` 后跟中文须空格。
- 依赖宏包：ctexart/beamer、amsmath、physics、slashed、natbib（无 biblatex）。
- 新文档放入本目录并登记到上方表格。

## 对照测试报告

- `report_cmp1_4150_20260828.tex/.pdf`：基于真实组态 4150 的 PyQCD 与 lqcddb/donghx 功能对照报告；16:9 横板、16 页，含 L20 复数 GEVP 差异判定、HYP 输入守卫、验证命令与未验证边界（2026-08-28）。
- `report_donghx_4150_reproduction_20260829.tex/.pdf`：基于真实组态 4150 的 donghx/PyQCD eigvec、VdV、VVV、Clover/dual/OPE 与质子 2pt 复现报告；明确 Cg5/Cg5g4、动量方向/大小、smear 输入边界及 3pt/ratio/barematrix 未验证项（2026-08-29）。
- `report_donghx_4150_reproduction_20260830.tex/.pdf`：在上一版基础上纳入 7 个 4150 2pt 成品根目录、58 个动量组、`momsmear0` 隐式命名检查及 Cg5/Cg5g4 各 35 个真实时间对；保留独立 smeared peram、逐时间 VVV 与下游 3pt/ratio/barematrix 的未验证边界（2026-08-30）。
- `report_donghx_pyqcd_compare_20260830.tex/.pdf`：简要汇总组态 4150 的已通过 dhx/PyQCD 低层、4D10 HYP-OPE、质子 2pt、正宇称投影与 effmass 聚合对照；表格列出数据路径、物理公式、PyQCD 函数/参数和相对误差（2026-08-30）。
- `report_donghx_pyqcd_compare_20260830_2.tex/.pdf`：同一对照结果的单一分页巨大表格版；仅保留已测试条目（2026-08-30）。
- `report_donghx_pyqcd_compare_20260830_2.csv`：对应巨大表格的 UTF-8 CSV；34 条已通过记录，含路径、公式、PyQCD 参数与相对误差（2026-08-30）。
