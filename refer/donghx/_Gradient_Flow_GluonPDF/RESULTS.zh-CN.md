# 新500副本裸矩阵元：线性发散减除

2026-11-22。unpolarized connected u-d；原配置120/100/70；22个flow，1855点，每点500副本。

## 减除定义

$M_{sub}(z,t)=\exp[+d_F|z|/r_f]M_{bare}(z,t)$，$r_f=a\sqrt{8\tau_{actual}}$。

只去除单根基础表示直线质量d_F/r_f；无额外2或Casimir。保留H3的k3及截距，未做z=0归一化、Zchi或matching。

|ensemble|d_F ± 原JK统计误差|拟合flow标签tau64|原矩阵元质量通过/总数|
|---|---|---|---|
|C24P29|0.387513633 ± 0.000909600|[0.5, 0.6]|112/448|
|E32P29|0.369198465 ± 0.000619738|[0.4, 0.5, 0.6]|147/672|
|F32P30|0.342325257 ± 0.001212458|[0.4, 0.5, 0.6]|177/735|

tau64=t_f/(0.0897 fm)^2是共同物理flow标签；实际tau=t_f/a²和rf逐flow保存。

## 误差与限制

- 误差棒是固定d后完整500行的SD(ddof=1)，不除sqrt(500)，不裁剪长尾，不重新拟合或重抽样。
- 阴影仅改变d±其原JK统计误差，不是总误差，也未包含pair/窗口/模型系统误差。
- Wilson系数JK与矩阵元bootstrap没有共同重采样索引；有配置交叠，不假设独立，不伪造逐行配对。
- 原436/1855中心质量通过，1419诊断回退；全部质量及replica标志原样保留。C24非零动量仍全为诊断。
- z=0数值与全部副本精确不变；7/8/9三个锚点的逐行连接恒等式已核验。
- C24系数来自最大两flow，存在显著pair依赖，基础三flow失败。F32是放宽门、三长度窄窗探索系数，内部留一flow训练零自由度。
- 星号为不在系数拟合flow域的外推；短距离及a/rf不小的点仍仅是诊断。所有系数final_coefficient_approved=false。
- 原T/a：C24为8–10，E32为9–12，F32仅11、12；F32的13、14没有输入。
- 单z方向、四current的普通flowed quark裸量；减除不等于完整重整化，不保证残余flow依赖消失。

## 图与数据

- `plots/<ensemble>_P*.png/.pdf`：每动量裸量/减除量的Re/Im对照。
- `plots/<ensemble>_all_momenta.png/.pdf`：每ensemble全部动量。
- `plots/three_ensemble_P4_all_flows_Re.png`、`Im.png`：共同整数P=4概览，非同物理动量比较。
- `plots/three_ensemble_P4_coefficient_fit_flows_Re.png`、`Im.png`：仅系数实际拟合的flow。
- 主图Re[-0.5,2]、Im[-2,0.5]、rf图例、横坐标轻微错开；全部数据不截断。超出固定y轴的点和误差请看`plots/autoscale/`。
- `matrix_elements.csv`：全部前后数值、SD、系数变化与质量；各flow `matrix_elements_subtracted.npz`：完整500行及全部原字段。
- `verification.json`、`tests.json`和`complete.json`：计算/回读/哈希验收，不代表物理质量全部通过。
