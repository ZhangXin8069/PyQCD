# 本次统计量与生产验收范围（2026-11-01）

本次“50 conf / 20 conf”明确指下面两批 B 方案 quark C3/C2，**不代表扩展到100/50的后续批次也已完成**。
最终分析完成状态请以本目录 `complete.json`、`DELIVERY.md` 和两个子目录的 `complete.json` 为准。

| Ensemble | 配置范围 | 已齐的动量 | 已齐的 flow |
|---|---|---|---|
| L32x64 | 10000..11960，步长40，共50 | P0,+/-4,+/-5,+/-6 | tau=0.1..0.8，步长0.1 |
| L32x96 | 10000..11100 与 12300..13000，步长100，共20 | P0,+/-3,+/-4,+/-5,+/-6 | 与 L64 tau=0.2..0.8 同物理半径的七点 |

L64 P3只在原前20配置中存在，不混入本次50配置产品。两套配置列表不会为了形成连续编号而擅自添加/移除配置。
两套数据都是4个 current 锚点、T=7..12、每 T 的全部 dt=0..T、z/a=0..20。

## 完成证据

- L64 原前20 moving：`../quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2/complete.json`，
  其配置级投影/ledger已包含midpoint正确复用路由。
- L64 新30 moving：`../quark_l32x64_B_expand50_P456_20261026/summary/complete.json`，
  `COMPLETE_30_NEW_CONF_8_FLOW_B_P456`，60480个case。
- L64 合并50 moving：`../quark_l32x64_B_ratio50_boot200_plateau_20261030/complete.json`，
  已完成200-bootstrap ratio和独立gather检查，尚未做本次裸拟合。
- L64 完整50 P0：`../quark_l32x64_B_Pz0_fill50_8flow_20261029/summary/complete.json`，
  `PASS_50CONF_8FLOW_B_PZ0`，100800个case；对应retry完成记录也已存在。
- L96 原20：`../quark_l32x96_B_matched_radius20_7flow_retry_v2_20261021/summary/complete.json`，
  `COMPLETE_L32X96_B_MATCHED_RADIUS_C3_C2`，35280个case、140个conf-flow任务。
- L96 已完成100-bootstrap分析及独立复核：
  `../quark_l32x96_B_ratio20_boot100_T8_rfmatched_20261029/complete.json` 和 `postcheck_v2/verified.json`。

本次 CPU gate 3886071_0/1 均退出0：两个完成批次的来源、源码身份、全部紧凑投影SHA和NaN/形状/配置一致性通过；
200次配置gather复数ratio、gamma、GLS和AIC/补选的独立小规模数值测试也均通过。
随后逐flow检查 ledger 中全部raw文件仍存在，并逐配置重读4锚点作独立投影oracle；L64 P0额外全部SHA/Wick/gamma重检。

## 新分析 CPU 作业

| 步骤 | L32x64 | L32x96 |
|---|---|---|
| 双ensemble前置gate | 3886071_0 | 3886071_1 |
| 配置bootstrap200及raw复核 | 3886072_[0-7] | 3886075_[0-6] |
| Ratio-AIC裸拟合（每flow打包9个通道/味） | 3886073_[0-7] | 3886076_[0-6] |
| 绘图与独立逐样本AIC/补选验证 | 3886074 | 3886077 |
| 两边完成后的总交付 | 3886078 | 3886078 |

所有科学数值计算均在CPU分区进行，未在登录节点统计/拟合，也未新增GPU生产。
第一版CPU gate仅在写JSON时遇到NumPy bool序列化错误，未进入ratio或拟合。
其后续依赖作业3886060/61/62/63已取消；错误日志保留在不带`_v2`的同名目录，勿误认为原始生产失败。
第二版修正Python bool类型并按flow打包，未更改物理/统计处方。

## 明确未并入的后台计算

检查时，L64第51..100配置（12000..13960）的3884031仍有GPU任务运行/排队；
L96扩展到50配置的3879283也仍有运行/排队任务。这些不属于本次前50/前20统计。
历史3863496的旧audit依赖无法满足，不覆盖新的retry正式完成证据；本次没有修改/取消它或任何后台GPU作业。

## 裸结果使用限制

按每ensemble原配置列表有放回抽样200次，seed1115；各flow/通道/味/P/z/T/dt和P0共享索引，两个ensemble不逐行配对。
ratio先分子/复数C2各自配置均值后相除，再±z和±P对齐。不是配置ratio的平均，也不是对拟合值二次bootstrap。
拟合Tmin为L64=7、L96=8，严格大于0.6fm；模型R=M；Q>=1e-3、chi2/dof<=1e3。
质量包络与最小误差诊断补选分别保存；200个bootstrap并非200个独立配置。
未做局域归一化、Zchi、线减除、flow matching或任何物理PDF重构。
