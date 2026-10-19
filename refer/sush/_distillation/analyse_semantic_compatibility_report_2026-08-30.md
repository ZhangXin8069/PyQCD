# 更新前后分析函数的作用与物理语义兼容性报告

## 1. 结论

更新后的函数保持了原函数的科学用途和目标物理量。没有发现意外改变符号、归一化、Fourier 相位、时间方向、ratio 公式、有效质量公式或 GEVP 方程的回归。

“相同”按以下三类判定：

1. **数值等价**：旧实现本来正确的输入域，新旧结果逐元素相同（仅允许浮点求和次序的舍入误差）；
2. **物理等价、标签修正**：数值代表同一物理时间/态，但旧数组位置或态标签不正确；
3. **有意不等价**：旧结果违反明确统计或物理定义，新实现不得复现旧错误。

新增 `test_analyse_semantic_compatibility.py` 的 8/8 项以及 `test_lazy_api.py` 的对象同一性检查均通过。

## 2. 逐函数结论

| 函数 | 兼容类型 | 证据 |
| --- | --- | --- |
| `Mom2GeV` | 数值等价 | 保留 $E=\sum_i\sqrt{(2\pi\hbar c/(N_xa))^2|\mathbf n|^2+m_i^2}$；多动量/多质量直接参考通过。新增内容仅为物理输入域校验。 |
| `loop_tsrc` | 数值等价 | Periodic/Antiperiodic 的 2pt 与固定 `t_sep` 3pt 在旧合法域逐元素相同；新增 `boundary_phase` 只是把完整 correlator 的绕回相位显式化。 |
| `sum_over_array_of_list` / `mean_over_array_of_list` | 数值等价 | 合法 grouping 的结果未改变；仅新增空组、重复轴和非法索引拒绝。 |
| `Jackknife` | 数值等价 + 有意修正 | 实数据、配置轴 0 的 sample/mean/error 与旧实现相同；covariance 最大差 $6.94\times10^{-18}$，来自 einsum 求和次序。复数据第二个 residual 加共轭、非首配置轴广播修复属于有意物理修正。 |
| `Bootstrap` | 同一估计量分布 + 有意修正 | 同一 RNG seed 下，新 replicate 0..11 与旧实现真正随机的 replicate 1..12 逐元素相同；新实现只是用一个真实 replicate 替换混入分布的中心估计量。默认 $M=N_{\rm conf}$ 时目标分布不变；`ddof=1`、m-out-of-n 缩放和原样本中心值是统计修正。 |
| `meff(log/GEVP)` | 数值等价 | 所有有效 $t$ 的 $\log[C(t)/C(t+1)]$ 值相同。 |
| `meff(cosh)` | 物理等价、时间标签修正 | 旧位置 $x$ 的值物理上属于中心时间 $x+1$；新实现把相同数值存到正确的 $t=x+1$。端点由伪物理 0 改为 NaN+mask。 |
| `ratio_3pt` | 数值等价 + 有意修正 | 正定、同轴、$0\le\tau\le t_{\rm sep}$ 域与旧公式逐元素一致；异轴对齐、复平方根和无效点 mask 修复旧广播/截断问题。二维不等初末能量合成平台恢复 1.7，残差 $<2\times10^{-14}$。 |
| `solve_gevp` | 同一广义本征问题、态标签改进 | 每个返回对满足 $C(t)v_n=\lambda_n C(t_0)v_n$，残差 $<2\times10^{-12}$，且 $v_n^\dagger C(t_0)v_n=1$。无 crossing 谱与旧独立排序相同；crossing 处只改变态标签连续性，不改变本征值集合。`track_states=False` 可显式恢复旧排序。 |
| `dis_connect` | 同一目标关联、旧轴 bug 有意修正 | 仍计算 $\sum_{t_{src}}\delta C_2\,\delta B$，没有把物理乘积误改为统计共轭 covariance。新结果与显式绝对→相对时间循环一致；旧 reshape/roll 错误不应保留。 |
| `phase_exp_VdV` / `phase_exp_VVV` | 数值等价重命名 | 都保持 $e^{-i\mathbf p\cdot\mathbf x}$；VDV 仅多颜色复制轴。直接坐标公式残差 $<2\times10^{-15}$。 |

## 3. lazy API 对物理对象的影响

lazy API 没有引入代理计算或包装公式：

- 顶层 `lqcddb.__all__` 中每个对象都与其实现模块的对象满足 Python `is` 同一性；
- `base/constant/analyse/contraction/eigvectors/io` 各子包同样满足对象同一性；
- 因此 named import、subpackage import 和实现模块 import 调用的是同一个函数/类，不存在多份实现或物理分叉；
- wildcard 只改变 optional API 的发现集合，不改变显式 named API；2-rank 顶层 named MPI `Allgather` 已通过。

## 4. 有意的输出差异

以下差异不能回退，否则会恢复非物理/非标准行为：

- 复 covariance：$r_i r_j\to r_i r_j^*$；
- Bootstrap：原均值不再伪装成 replicate 0；
- m-out-of-n：误差乘 $\sqrt{M/N_{\rm conf}}$；
- `meff`/`ratio_3pt` 域外点：0 改为 NaN+`valid_mask=False`；
- ratio 负平方根：截断 0 改为复主值；
- cosh 结果位置：从窗口左端改到物理中心；
- GEVP crossing：独立大小排序改为本征矢连续跟踪；
- 非正定 $C(t_0)$、非法轴/shape：继续计算改为 fail closed。

这些变化保持或恢复目标物理定义，但不保证依赖旧错误数组位置、dtype 或 sample-0 约定的下游脚本无需调整。

## 5. 验证汇总

- 兼容性差分：8/8；
- analyse 独立物理测试：10/10；
- lazy API：5/5；
- 其余串行回归：46/46；
- 总计：69/69；
- 50 个 `.py/.pyi` AST：0 错误；
- 112 个 callable mapping 与 `.pyi`：0 签名不一致；
- 当前源码/存根/测试聚合 SHA-256：`48d59cb5b68269b9b1ef209ace6597a1e9f8df6ccba6109fa19affcae2d60c0e`。

## 6. 未声称兼容的范围

本结论只覆盖本轮更新的 analyse、lazy API、存根、调用点与相位方法重命名。Stout、Wick/dynamic 图筛选、等价图 flavor 假设、conserved temporal VDV、eigenvector/omega、性能 adviser 和两个既有 MPI 布局问题仍按此前报告处理。
