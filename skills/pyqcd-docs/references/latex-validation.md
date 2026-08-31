# LaTeX 与 PDF 验收 reference

## 编译

在包含源文件的目录执行两遍 XeLaTeX：

```bash
cd docs
xelatex -interaction=nonstopmode <document>.tex
xelatex -interaction=nonstopmode <document>.tex
```

若项目已有报告构建脚本，优先调用脚本并记录命令；不要为了验证在源目录生成未登记
的缓存、图片或辅助文件。中文文档不用 `pdflatex` 替代 XeLaTeX。

### 报告管线的运行时硬门

对 `pyqcd/pipeline/_steps.py::step_report`，上述通用验收之外还必须满足：

- 两遍必须是实际 `xelatex` 调用，且每一遍 `returncode == 0`；启动失败或任一遍非零都失败，不能用模拟调用或旧结果替代。
- 编译前已有的 `physics_report.pdf` 不能单独证明成功。成功要求 PDF 存在且相对编译前文件签名发生变化；签名包含 `mtime_ns`，因此未更新 `mtime` 的 stale PDF 必须失败。
- 三类诊断在任一遍的捕获 `stdout`、`stderr` 或该遍 `physics_report.log` 中出现，都必须失败；不能只检查最后一份日志。

## 日志硬闸门

两遍日志都必须逐项确认：

```bash
rg -n "Overfull|Float too large|Missing character" <log-1> <log-2>
pdfinfo <document>.pdf | rg "Pages"
```

`Overfull`、`Float too large` 或 `Missing character` 任一出现都不是“可忽略警告”。
修复断行、列宽、字体或浮动体后重新两遍编译；不要只检查退出码。

## 逐页检查

用本机可用的 PDF 渲染工具把每页导出到本目录之外的临时目录，逐页检查：

1. 页面尺寸和总页数与交付记录一致；
2. 标题、正文、代码和数学字体可读，中文/希腊字母没有方框；
3. 图表、公式和页眉页脚位于安全区，没有裁切、重叠或遮挡；
4. 图题能说明物理意义、对应结果和解释，不能只放截图。

检查后删除或移走临时渲染物，最终目录只保留登记的源、PDF 和必要日志。若工具不可用，
必须在报告中注明“未完成逐页渲染”，不能把静态 log 检查写成版式验证。

## 字体与溢出修复优先级

先改内容断行和布局，再调整图表尺寸；等宽代码优先使用覆盖源码 Unicode 的字体（如
DejaVu Sans Mono）。禁止用 `\tiny` 或 `\scriptsize` 作为通用溢出修复；只有明确的
局部外部规范要求时才例外，并记录原因。
