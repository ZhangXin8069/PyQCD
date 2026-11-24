# 本轮结果从这里开始

**本轮已完成数值及绘图，并通过独立回读。优先看 [DELIVERY_GUIDE.md](DELIVERY_GUIDE.md)：它包含各阶段代表图的直接链接、统计与拟合约定、连续极限做法及未通过的质量检查。**

1. **进度**：RUN_STATUS.md；用 `python -B status.py` 更新，只查元数据和Slurm，不读科学数组。
2. **参数/估计量/理论与限制**：README.md、00_audit/KNOWN_SYSTEMATICS.md。三个ensemble为100/40/30conf，各200bootstrap。
3. **真实裸拟合与每步数值**：L32x64/、L32x96/、L24x72/，三格距数值在continuum/。冻结数值源码在code/，input_plan/source_lock明确来源。数值不因绘图修复重算。
4. **正确图片交付根**：plot_repair_v2_20261108/plots/。父目录旧plots/是初版色表越界留下的部分图，不能当完整交付。修正版保留实部[-1,2]、虚部[-2,1]和不连线误差棒。
5. **最终验收**：父目录complete.json只有修正版全部PDF和数值独立回读通过后才发布。完整验收表位于plot_repair_v2_20261108/00_audit/，PLOTS.md生成后列出所有PDF。
6. **质量而不只是覆盖**：00_audit/interpretation_v3/SUMMARY.md、summary.json、bare_replica_fallback.csv、continuum_quality.csv、flow_quality.csv、m0_primary.csv、model_sensitivity.csv。CSV/JSON由独立CPU汇总作业生成。
   加权汇总见 `00_audit/delivery_statistics_v1/SUMMARY.md`；中心回退与至少一行回退必须区分。
7. **代表性页面渲染**：plot_repair_v2_20261108/00_audit/visual_review/。渲染完成不等于物理验收。
   最终成功版本为 `plot_repair_v2_20261108/00_audit/visual_review_v5/`；见 `VISUAL_REVIEW.md`。此前无版本号目录是失败尝试，不当验收。

主支固定物理流时间、z、P做a²线性外推，再做纯线性小流时间外推和P0的诊断m0。共同P=0,2.0,2.25,2.5GeV；z=0,.105,...,1.470fm；tf对应tau64=.2..8。
先前固定a的05/06另存控制分支，不能冒充连续极限。D_loose缺失不补零，不借allflow伪装受控结果。

**全部有限数值/200行覆盖不等于质量通过；固定dF候选、质量/体积/激发态/强流窗、完整MSbar方案与transversity外部符号仍未闭合。**

CPU作业当前编号及历史失败/替代记录：submission.json、plot_repair_v2_20261108/submission.json、postcheck_submission_v2.json。
原绘图失败仅为第5个动量访问第9种颜色，修正版增加为9色并使用独立输出目录；旧依赖final3954963/visual3955047已取消，GPU生产任务未改动。

最终数值回读作业3955054正常完成：2,844个数值产品，270个PDF。原环境不含PDF解析器，因此另用本目录私有的固定PyMuPDF轮子做CPU后置页数和预览验收；记录见 `postcheck_submission_v3.json`。这不改变任何科学数组和数值源码。
后置作业3955107正常完成，全部PDF回读通过，共1,647页；16张预览中已目视抽查8张代表页，质量限制仍按原记录保留。

生成日期：2026-11-08（Asia/Shanghai）。会话标题：三ensemble quark B方案、200bootstrap与探索性连续极限。
