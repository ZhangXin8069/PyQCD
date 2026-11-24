# 两个 ensemble 的 B 方案：200-bootstrap 裸矩阵元

## 已完成的范围

| Ensemble | Nconf | Nboot | Pz | Tmin | 有限结果 | 无补选质量点 | 含诊断补选 |
|---|---:|---:|---|---:|---:|---:|---:|
| L32x64 | 50 | 200 | [0, 4, 5, 6] | 7 | 12096/12096 | 194 | 11614 |
| L32x96 | 20 | 200 | [0, 3, 4, 5, 6] | 8 | 13230/13230 | 9 | 12906 |

固定模型为 $R(T,t)=M$；$Ta>0.6$ fm；$Q\ge10^{-3}$、$\chi^2/\mathrm{dof}\le10^3$。
每个 bootstrap 独立执行 cut 内 AIC 窗口加权，再等权混合可用 cut。无通过候选时的最小统计误差补选单独标记。
图中实心点为不依赖补选的质量结果；空心点为诊断补选/质量不完整；叉号为 odd z0 代数零（未拟合）。
不能将有限覆盖率或空心点升级为可靠的平台信号。

## 图片和数据入口

### L32x64

- [验收与质量汇总](L32x64/RESULTS.md)
- [所有逐点结果](L32x64/bare_matrix_progress.csv)
- [按通道/味/动量分组的质量覆盖率](L32x64/coverage_by_channel_flavor_momentum.csv)
- `./L32x64/ratios/`：200-bootstrap 复数 ratio。
- `./L32x64/fits/`：裸矩阵元、200 样本、质量/补选和全部候选。

- [unpolarized u](L32x64/plots/bare_unpolarized_u_allflow_boot200.pdf)
- [unpolarized d](L32x64/plots/bare_unpolarized_d_allflow_boot200.pdf)
- [unpolarized u_minus_d](L32x64/plots/bare_unpolarized_u_minus_d_allflow_boot200.pdf)
- [helicity u](L32x64/plots/bare_helicity_u_allflow_boot200.pdf)
- [helicity d](L32x64/plots/bare_helicity_d_allflow_boot200.pdf)
- [helicity u_minus_d](L32x64/plots/bare_helicity_u_minus_d_allflow_boot200.pdf)
- [transversity u](L32x64/plots/bare_transversity_u_allflow_boot200.pdf)
- [transversity d](L32x64/plots/bare_transversity_d_allflow_boot200.pdf)
- [transversity u_minus_d](L32x64/plots/bare_transversity_u_minus_d_allflow_boot200.pdf)

### L32x96

- [验收与质量汇总](L32x96/RESULTS.md)
- [所有逐点结果](L32x96/bare_matrix_progress.csv)
- [按通道/味/动量分组的质量覆盖率](L32x96/coverage_by_channel_flavor_momentum.csv)
- `./L32x96/ratios/`：200-bootstrap 复数 ratio。
- `./L32x96/fits/`：裸矩阵元、200 样本、质量/补选和全部候选。

- [unpolarized u](L32x96/plots/bare_unpolarized_u_allflow_boot200.pdf)
- [unpolarized d](L32x96/plots/bare_unpolarized_d_allflow_boot200.pdf)
- [unpolarized u_minus_d](L32x96/plots/bare_unpolarized_u_minus_d_allflow_boot200.pdf)
- [helicity u](L32x96/plots/bare_helicity_u_allflow_boot200.pdf)
- [helicity d](L32x96/plots/bare_helicity_d_allflow_boot200.pdf)
- [helicity u_minus_d](L32x96/plots/bare_helicity_u_minus_d_allflow_boot200.pdf)
- [transversity u](L32x96/plots/bare_transversity_u_allflow_boot200.pdf)
- [transversity d](L32x96/plots/bare_transversity_d_allflow_boot200.pdf)
- [transversity u_minus_d](L32x96/plots/bare_transversity_u_minus_d_allflow_boot200.pdf)

每份 PDF 两页：第 1 页完整 y 范围；第 2 页实部 [-1,2]、虚部 [-2,1] 的固定视窗。
所有 z/a=0..20 均保存；不同 flow 的 x 坐标略错开，误差棒不连线。

L32x64 保留 tau=0.1..0.8 全部已算 flow（其中 tau=0.1 不满足此前 a64<rf 的半径窗口，
本次只做裸拟合，未以此删点）。L32x96 为与 L32x64 tau=0.2..0.8 相同物理流半径的七点。
L32x64 的 Pz=3 仅有原前20conf，未混入50conf结果。两个 ensemble 的 bootstrap 行不能逐行相减作配对差。

本次未做局域归一化、ringed Zchi、线减除、gradient-flow matching、flow-time/连续极限外推或 LaMET。
connected u/d 非完整 singlet PDF；transversity 相对符号仍待外部物理定义确认。

生成时间：2026-11-01T13:52:16.241520+08:00。
