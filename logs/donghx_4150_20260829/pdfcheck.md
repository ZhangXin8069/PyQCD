# 4150 对照报告 PDF 验收

报告：`docs/report_donghx_4150_reproduction_20260829.pdf`

## 编译

在仓库内隔离目录 `.pdfcheck.4FppeV/build-next/` 执行 XeLaTeX 两遍：

```
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=../.pdfcheck.4FppeV/build-next \
  report_donghx_4150_reproduction_20260829.tex
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=../.pdfcheck.4FppeV/build-next \
  report_donghx_4150_reproduction_20260829.tex
```

结果：

```
exit code: 0
pages: 23
page size: 453.54 x 255.12 pt (16:9)
Overfull: 0
Float too large: 0
Missing character: 0
LaTeX Warning: 0
```

## 渲染与目检

使用 `pdftoppm -png -r 120` 渲染得到 23/23 页；联系页为：

```
.pdfcheck.4FppeV/render-next/contact-01.png
.pdfcheck.4FppeV/render-next/contact-02.png
.pdfcheck.4FppeV/render-next/contact-03.png
```

23 页均已覆盖检查；新增的 momentum-smear 输出级页、结论页、smear 边界页和
验收页另以原尺寸复核，未见裁切、遮挡、越界或不可读表格。
