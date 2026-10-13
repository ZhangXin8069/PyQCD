# 三点矩阵元方法 reference

## 当前状态总览

| 方法 | PyQCD 状态 | 可调用入口 |
|---|---|---|
| ratio plateau | 已实现并用于连通 3pt/2pt | `pyqcd.analysis._analyse.ratio_3pt`、`pyqcd.analysis._correlators.run_3pt_ratio` |
| 02_ratio OPE/2pt 生产链 | 已实现 | `pyqcd.analysis._ratio2pt.run_ratio2pt` |
| summation | 未落地，仅公式模板 | 无 |
| C2+C3 联合二态 | 未落地，谱式在 `pyqcd-physics-spectrum` | 无 |

“存在谱式或公式”不等于有生产入口。新增调用前先用 `rg` 和 import 核实当前源码，
不要为本表中标为“未落地”的方法虚构模块名。

## Ratio plateau

数值基元 `ratio_3pt` 对带组态轴的样本计算

\[
R(\tau)=
\frac{C_3(\tau)}{C_2^F(t_{\rm sep})}
\sqrt{
\frac{C_2^I(t_{\rm sep}-\tau)C_2^F(\tau)C_2^F(t_{\rm sep})}
{C_2^F(t_{\rm sep}-\tau)C_2^I(\tau)C_2^I(t_{\rm sep})}
}.
\]

当前一维或二维时间轴模式由 `t_src_axes` 选择，`tau_axes` 与 `t_sink_axes` 必须显式
对齐。若初末态相同且取零动量弹性运动学，上式退化为常见的 \(C_3/C_2\)；生产代码
仍应按函数签名传入明确轴，不能依赖广播碰巧正确。

现有生产路径：

1. `pyqcd.analysis._analyse.ratio_3pt`：底层 ratio 计算，输入 jackknife 后的 3pt/2pt
   样本。
2. `pyqcd.analysis._correlators.run_3pt_ratio`：从组态字典提取 `proton_P*_3pt` 的
   \(\gamma_3\) 分量并调用底层 ratio；返回 `R/R_err/t_sep`。
3. `pyqcd.pipeline._steps.step_analysis`：管线有 3pt 数据时调用
   `run_3pt_ratio`，并落盘 `ratio_<had>_<mom>_{mean,err}.npy`。
4. `pyqcd.analysis._ratio2pt.run_ratio2pt`：02_ratio 的 OPE/2pt 生产链，完成真空扣除、
   逐 `z` 拟合及 `c0/chi2` 图；不是 generic 3pt current 的通用拟合器。

可运行示例入口（第二个是 test0 合成数据回归，不是真实系综结果）：

```bash
python -m pyqcd.testing.spectrum.effective_mass_ratio
bash data/from-logs/logs/test0_ratio/run-local.sh
```

## Summation：仅模板

若已有多个 \(t_{\rm sep}\) 的可靠 ratio，可定义

\[
S(t_{\rm sep})=a\sum_{\tau=t_0}^{t_{\rm sep}-t_0}R(\tau),
\]

并在激发态污染可忽略区对 \(t_{\rm sep}\) 做线性拟合，斜率为目标矩阵元。当前 PyQCD
没有 summation 公共函数或生产入口；本式只能作为未来实现的契约，不能写成现成 API。

## C2+C3 联合二态：仅谱式模板

三点函数可写成

\[
C_3(\tau,t_{\rm sep})=
\sum_{n,m}B_{nm}
e^{-E_n^f(t_{\rm sep}-\tau)}e^{-E_m^i\tau},
\]

其中两侧 \(E_n,Z_n\) 应与 C2 联合拟合共享，基态矩阵元由
\(B_{00}=Z_0^f\langle0|J|0\rangle (Z_0^i)^*/ (4E_0^fE_0^i)\) 或同约定关系恢复。
谱式、backward 项和参数化以 `pyqcd-physics-spectrum` 为权威来源。

当前没有 C2+C3 joint-fit 的生产模块、结果 schema 或回归入口。交接时必须把“谱式
模板”和“已实现拟合”分开，且不能用三点振幅直接冒充矩阵元。

## 必查边界

- 3pt、2pt 和 OPE 使用同一组态索引、源时刻、动量/Fourier 符号及 \(t_{\rm sep}\)。
- `tau`、sink time、source time 和 link/`z` 轴必须显式指定并进行 shape 断言。
- 保存 contact-term 排除范围、拟合窗口、协方差/SVD、`Ncfg` 和所有无效点 mask。
- 取实部、fold link 或奇偶组合前必须有宇称、共轭或显式投影依据。
- 用合成解析数据检查 ratio 归一化和矩阵元恢复，再对真实数据报告系统误差。

来源参考：`skills/LQCD_Skills/skills/lqcd-analysis/SKILL.md`，submodule 提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`；实现状态已按
`pyqcd.analysis` 与 `pyqcd.pipeline._steps` 当前源码逐项核实。
