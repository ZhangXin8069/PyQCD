# AGENTS.md — docs

格点 QCD 中文 LaTeX 与分析/报告文档目录。当前工作树包含 11 个 TeX 源文件和
9 个 Markdown 文件；LaTeX 文档统一使用 xelatex 编译。`docs/.gitignore` 禁止
新增嵌套目录，历史迁移材料的当前事实以 Git 索引为准。

## 文件

| 类别 | 文件（节选） |
|---|---|
| 全库/工作流 | `analy_pyqcd_20260818.tex`、`pure_pyqcd_20260818.tex`、`analy_physics_chain_20260824.tex`、`analy_torch_parallel_20260816.tex` |
| 谱学/专题 | `analy_proton_energy_20260816.tex`、`analy_gpd_20260928.tex` |
| TMD/重整化报告 | `report_gluon_tmd_gradient_flow_20260828.tex`、`report_tmd_gluon_audit_20261025.tex`、`report_gluon_tmd_flow_scan_20261029.tex` |
| 参考对照 | `report_cmp1_4150_20260828.tex`、`report_gradient_flow_gluonpdf_pyqcd_20260909.tex` |

TeX 编译的中间文件不纳入结果证据；提交文档时只保留源文件与需要交付的 PDF。

`examples/` 目录已于 2026-09-27 整体迁移到 `pyqcd/testing/`；历史报告中的旧
路径映射见 `docs/EXAMPLES_MIGRATION_20260927.md` 与
`pyqcd/testing/MIGRATION.md`。

## 编译

```bash
cd docs && xelatex -interaction=nonstopmode <文件>.tex   # 中文必须 xelatex；两遍
```

## 约定

- 文件名统一中文（英文名已迁移时中文化）；tex 内路径引用指向 `/root/PyQCD/`。
- 文本模式数学符号需 `$...$` 包裹（如 `$\pi$介子`）；`\quad` 后跟中文须空格。
- 依赖宏包：ctexart/beamer、amsmath、physics、slashed、natbib（无 biblatex）。
- 新文档放入本目录并登记到上方表格。

## 分析经验记录

- `gluon_tmd_analysis_notes_20261029.md`：梯度流胶子准 TMD 比值分析的可复用经验
  （复数比值取实部、流窗与空集合判定、配对 JK 与检验功效、9 组态协方差秩、
  零成本 tsep 扩展、复用已发布拟合、数据审计与 LaTeX 版式），对应报告
  `report_gluon_tmd_flow_scan_20261029.tex`（2026-10-29）。

## 格式治理

- `form_pyqcd_20261027.md`：记录 2026-10-27 胶子 TMD-PDF 诊断交付的格式审计结果；
  `git diff --check` 与脚本语法通过，3 项既有仓库问题保留为治理债。
- `form_pyqcd_20260928.md`：记录 PyQCD 命名/目录/归档边界审计、具名例外、本地
  `form` 审计入口和整改前后功能复现证据。

## 对照测试报告

- `report_cmp1_4150_20260828.tex`：真实组态 4150 的 PyQCD 与 lqcddb/donghx
  功能对照，含 GEVP、HYP 输入守卫与未验证边界。
- `report_gradient_flow_gluonpdf_pyqcd_20260909.tex`：Wilson flow、Clover/dual、
  schema-v2 OPE 与 disconnected ratio 的算法审计。
