# 数据、生产门与验收措辞

## 1. 参考轴序

历史参考工作流中常见对象：

```text
flowed basis: (eigen,z,y,x,color) = (400,32,32,32,3)
vertex:       (vdv_sign,delta_z,eigen_left,eigen_right) = (2,21,400,400)
corr_2pt:     (pol,mom,t_source,t_sink)
corr_3pt:     (sign,pol,mom,dz,current_m,current_o,t_source,t_current)
ratio_v4:     (boot,pabs,delta_z,tsep,tau)
```

`ratio_v4` 的公共 `tau` 轴必须配套 `ratio_valid_mask`；非物理 padding 和分母拒绝项
保存为 NaN，不能填零。所有方向、偶奇通道、flavor、动量、`z` 和方法共享同一
`boot_indices`。

ringed paired 产品至少保存：

```text
flow_times, confs, bootstrap_indices
kinetic_noise: (flow,conf,noise)
kinetic_config: (flow,conf)
Zchi, Zchi_error, Zchi_bootstrap, Zchi_covariance
bare_central, ringed_central, ringed_error
bare_bootstrap, ringed_bootstrap
fit_valid, fallback/status fields
sfte_* diagnostic fields
small_flow_window_mask, formal_result_valid
```

轴和字段名可能随版本变化；先读实际 manifest/schema，不按旧目录名猜含义。

## 2. 生产前检查

1. 确认 PyQUDA/QUDA 真暴露 gauge flow 和 Dirac-Laplace，而非只暴露接口名。
2. zero-flow 与未流 VDV/C2/C3 对照。
3. `V^dag V`、color order、轴序和 stochastic weights。
4. flowed link 的幺正性、行列式、plaquette/action 和时间相位。
5. 自由场 heat kernel、`epsilon` 减半和直接/restart 对照。
6. 费米子 APBC 和 gauge covariance。
7. 时间窗口丢弃范数与更宽窗口的 C3 对照。
8. factorized contraction 与显式双时间和的小张量 oracle。
9. 单 rank/多 rank 的轴、归约、输出和 hash。

任何一项失败都停止把数据命名为 accepted；旧任务继续运行时不要原地修改其脚本。

## 3. 拟合前检查

- 只消费当前版本 `ratio_v4`；旧版本存在 C2 momentum-axis 错误时只能作为 traceability。
- 检查 even/odd 方向对齐；negative-momentum odd 方向先乘预期 parity 因子再比较。
- 固定物理 `Tmin`，候选 window 的 `Tmax` 才可扫描。
- 用同一组态、index、projection、方向平均阶段和 mask 比较 Ratio/Sumratio 方法。
- 对方法差异使用 paired bootstrap pull，不能把两个独立误差相加。
- 保存 quality-accepted 与 fallback 两套字段，不用 filled 数组覆盖 accepted 数组。

## 4. ringed 验证

- 全体积 Z4 noise 满足 `E[eta eta^dag]=I`，不能归一到单位向量。
- 导数、APBC、gauge array 和 fermion array 轴按实际实现核查。
- 复核真实 full-MAT defect，不只看 preconditioned residual 或退出码。
- 保存 original/recovery 代码、solver 参数、gauge/code hash 和每组态 noise 身份。
- 对虚部作预先声明的 zero-consistency 检查；不能凭中心值小就删除虚部。
- 对质量处方、noise 收敛、autocorrelation、solver 等价性和 perambulator provenance
  分别给出状态。

## 5. SFTE/matching 验收

- `Z_chi` 的 accepted JSON 必须有真实 hash 绑定和人工科学审查；模板不能升级。
- 每个 bootstrap replica 乘同一确定性 SFTE 系数，保留原 index。
- 保存 `mu`、`alpha_s`、`CF`、系数符号、`delta_m` 和窗口 mask。
- matching 前确认 GF-to-MSbar conversion 与 quasi-to-light-cone kernel 的完整
  scheme 不重复、不冲突。
- 逐 channel 检查 kernel；transversity 相对符号未外部核实时不得升级。

## 6. 状态措辞

| 数据层 | 可接受措辞 |
|---|---|
| raw C2/C3 或 ratio | finite-flow connected correlator / bare diagnostic |
| bare fit | unrenormalized connected matrix element |
| 有 Zchi 但无 SFTE | ringed finite-flow matrix element |
| 有 SFTE 无 LaMET | MSbar bilocal diagnostic |
| 有 matching 但无多尺度控制 | exploratory matched diagnostic |
| 有真实系综、受控极限和系统误差 | matched nonsinglet PDF |

`physical-validated` 需要真实系综、独立物理检查、完整误差账本和声明范围；单张图、
单次数值成功或与参考颜色看起来一致都不够。
