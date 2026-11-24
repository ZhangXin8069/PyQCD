# PDF 与版式复核（2026-11-01）

- 18份最终PDF均由外部 `pdfinfo` 打开成功，均为2页；CPU验收也检查了各PDF的SHA、header/EOF与Page计数。
- 实际渲染并目视检查了L32x96 unpolarized u-d的两页，及L32x64 helicity/transversity u-d的第一页。
- 检查了图例与子图无重叠、Pz分行、even-real/odd-imag分列、全部z/a=0..20、不同flow轻微错位、误差棒不连线。
- 第1页完整y范围保留小的非零虚部；第2页固定视窗不意味着虚部被丢弃。实际统计检验使用NPZ而非视觉近似。
- 预览PNG仅在 `/tmp/quark_B_boot200_preview.Pwsmi4/`，不是正式结果；正式PDF位于两个ensemble各自的`plots/`。
- 未逐页目视检查全部36页；PDF结构验收覆盖全部18份。版式通过不等于平台/物理质量通过，空心诊断补选点必须结合质量报告解释。

跨flow C2的最终独立数值界限及逐动量记录见`L32x64/complete.json`和`verification_v3/SUBMITTED.md`；
额外补选比例统计见`postcheck_v1/RESULTS.md`。根`complete.json`已记录两套完成标记的SHA。
