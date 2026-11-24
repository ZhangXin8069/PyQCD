# PDF 回读与抽页目视检查

日期：2026-11-08，Asia/Shanghai。会话：三ensemble B方案、200 bootstrap与探索性连续极限。

CPU 作业3955107正常完成；固定私有PyMuPDF=1.24.14，只安装到本轮 `tools/`，没有改现有环境或科学数组。
全部270份PDF均核对了SHA256、PDF可读性、非加密状态、正页数与首尾页文字；共1,647页。
8类PDF各渲染2页，共16张PNG。机器凭证见 [validation.json](plot_repair_v2_20261108/00_audit/visual_review_v5/validation.json)。

人工抽查了下列8张渲染图，未发现图例与图框重叠、标题裁切、数据点误连线或 moving 虚部被全部置零：

|PNG|来源/检查|
|---|---|
|00_page003.png|L32x64 unpolarized u-d 裸矩阵元，nP=5；8个flow的图例分行清晰，flow点错开，实部和虚部均保留。|
|01_page003.png|L32x96 helicity u-d，RGR转换后nP=4；7个共同物理半径图例清楚，moving虚部非零。|
|02_page003.png|L24x72 unpolarized u-d，nP=5的ratio与拟合带；T=7..12图例未覆盖数据；原ratio很噪，拟合带窄不等于基态已经受控。|
|03_page003.png|共同物理tf,z,P下三ensemble；4个共同P的两分量均显示，零动量虚部接近零不等于moving虚部被清零。|
|04_page001.png|a²拟合代表页；格点输入、拟合线/带和a=0截距可辨认，Q及dof列出；数据误差在固定全范围内很小，非绘图漏画。|
|05_page003.png|连续后纯线性tf拟合，P=2.25GeV；蓝色flow点、红色拟合及截距清晰，噪声较大处仍保留完整误差。|
|06_page002.png|helicity u-d，连续后D_loose flow0；短距空窗没有补零，moving虚部非零；高P误差很大。|
|07_page002.png|transversity u-d诊断m0修正后；两分量保留，质量警告标题存在；不能因为图有限而认为m0质量通过。|

所有抽查图的实部范围为[-1,2]，虚部[-2,1]。部分大误差或中心值被用户固定范围裁切，这是显示限制，不是把数据修改到边界。
数值账本另存各图点的裁切标志；拟合曲线允许连线，数据误差棒不连接。

这里只是图形与文件验收，不等于统计/物理验收，也不是声称已逐页目视检查全部1,647页。
完整质量结论见 [质量总表](00_audit/delivery_statistics_v1/SUMMARY.md) 和 [交付指南](DELIVERY_GUIDE.md)。
