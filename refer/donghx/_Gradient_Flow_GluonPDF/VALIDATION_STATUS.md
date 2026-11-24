# 初始验证记录

记录日期：2026-11-05，Asia/Shanghai。此为阶段记录，不代表最终结果已经完成。

- CPU 审计 job 3930030：COMPLETED，exit 0，运行 6 秒，cpu009。
  L32×64 的 1700 conf 与 L32×96 的 777 conf 清单、全部共享 200 bootstrap 索引及权重、
  seed=1115 和来源身份已核对；没有把旧拟合样本二次抽样。
- CPU 数值测试 job 3930031：COMPLETED，exit 0，运行 39 秒，cpu010。
  FFT 与显式配置/时间循环、直接 resampled-array 真空减除、实部优先与完整复数处方差异、
  paired self ratio、零及翻号分母保护、GLS/SS 协方差、AIC/fallback、物理 Tmin、绘图轴均通过。
- `validation/plot_style_fixture.pdf` 和 PNG 是明确标注 SYNTHETIC NOT DATA 的绘图测试，
  不是组态数据结果。已人工查看 PNG，图例和面板分离、纵轴固定、点不连线。
- 两个真实数据先导作业 3930032_4、3930032_18 已在 cpu009/cpu010 启动，
  正在从原始 OPE 重新构造 real-first C3；输入检查未见错误。

真实 ratio、分母稳定性、Ratio–AIC 覆盖率和图像需等先导/生产阶段完成后确认。
不能根据合成测试通过，就宣称实数据平台或基态隔离已得到物理验证。

## flow=0.5 真实数据先导更新

两个 ratio 子任务均已正常完成：

| ensemble | conf / bootstrap | ratio 运行时间 | 全部 U 组合稳定分母 | 主算符稳定分母 |
| --- | --- | --- | --- | --- |
| L32×64 | 1700 / 200 | 9 分钟 | 1085/1815 | 290/605 |
| L32×96 | 777 / 200 | 4 分 7 秒 | 1013/1815 | 291/605 |

主算符分母数是 `(P,T,insertion)` 格点数，不是已完成拟合的 z 点数。
分母门和完整拟合窗的可用性是两回事；Ratio–AIC 任务仍须检验固定窗口内所有点。

产物已经原子保存到各 ensemble 的 `01_real_first_c3/tau0p500.npz`
与 `02_self_ratio/tau0p500.npz`，并具有 SHA/完成凭据。
先导拟合 job `3930033_[4,18]` 当前等待 CPU 调度；尚未产生真实数据的拟合图。
后续任务已按依赖提交，拟合、绘图和全量验证会自动接续。
