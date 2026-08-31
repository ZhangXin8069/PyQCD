# Statistics identifiability reference

本 reference 是 `pyqcd-statistics` 对有限输入、协方差秩、模型 Jacobian、prior 状态和
可辨识性验收的唯一细节入口。入口技能只保留关键不变量；遇到近简并模型、`NaN/Inf`、
先验主导或状态字段问题时读取本文件。

## 有限输入与样本轴

### 当前重采样 API

```python
import numpy as np
from pyqcd.analysis import sem, resample

data = np.arange(12, dtype=np.float64).reshape(3, 4)
jackknife = resample(data, jackknife=True, axis=0)
jackknife_error = sem(jackknife, jackknife=True)

# Bootstrap-specific options; omitting seed preserves the historical seed=0
# sequence, while axis selects the configuration axis.
bootstrap = resample(
    data,
    jackknife=False,
    Nsample=8,
    axis=0,
    chunk_size=2,
)
bootstrap_error = sem(bootstrap, jackknife=False)
assert jackknife.shape == data.shape
assert bootstrap.shape == (8, 4)
assert jackknife_error.shape == bootstrap_error.shape == (4,)
```

`Nsample` 是 bootstrap replica 数；每个 replica 固定从 `Nconf` 个组态中有放回抽取
`Nconf` 次，当前没有独立的 resample-size 参数。bootstrap 必须显式
`jackknife=False, Nsample=<positive int>`；`chunk_size` 也必须是正的非布尔整数。例如：

```python
import numpy as np
from pyqcd.analysis import sem, resample

data = np.arange(2 * 120 * 4, dtype=np.float64).reshape(2, 120, 4)
boot = resample(
    data,
    jackknife=False,
    Nsample=600,
    seed=17,
    axis=1,
    chunk_size=8,
)
# data.shape == (2,120,4) -> boot.shape == (600,2,4)
error = sem(boot, jackknife=False)  # shape (2,4)
```

`sem` 消费的是重采样后的估计量，不会再除以 `sqrt(Nsample)`。`jackknife=True`
使用 population 标准差再乘 `sqrt(Nsample-1)`；`False` 直接返回 bootstrap
replica 的 population 标准差。NumPy/CuPy 固定 `ddof=0`，Torch 固定
`correction=0`，API 不提供改变它们的参数。对复数输入，定义为
`sqrt(mean(abs(x-mean(x))**2))`，因此 `complex64/128` 的 `sem` 分别返回
`float32/64`。

`rng` 必须是调用者持有的 `numpy.random.Generator`；传它时必须完全省略
`seed`，即使显式 `seed=None` 也算冲突并报错。不传 `rng` 时，省略 `seed`
等价于历史可复现的 `seed=0`；显式 `seed=None` 则交给 NumPy 产生非确定初始状态。
索引统一由 NumPy Generator 生成后适配到当前数组后端，因此同一 seed/初始
Generator 状态在 NumPy/CuPy/Torch 之间使用同一索引序列。无论是 seed 路径还是
Generator 路径，改变 `chunk_size` 都不改变拼接后的完整索引序列。

单组态 bootstrap 定义为把唯一组态重复为 `Nsample` 个 replica；由此得到的零
spread 不是有效统计误差证据。单组态 jackknife 与单样本 `sem` 返回保持 shape 的
`NaN`。jackknife 为保持旧位置 API 仍忽略 `Nsample/seed`，但会拒绝 bootstrap
专属的 `rng/chunk_size`。

对每个分块，当前索引为 NumPy `int64` 的
`(chunk_size,Nconf)`；Torch 还可能同时持有同形 host/device 索引。gather 中间量逻辑
shape 为 `(chunk_size,Nconf,*feature_shape)`，故三类主要内存分别为：

- 索引：`O(chunk_size*Nconf)`；
- gather：`O(chunk_size*Nconf*feature_volume)`；
- 最终输出：`O(Nsample*feature_volume)`。

这些是数组 payload 量级，不是 RSS/显存 allocator 的严格上界。

- `sem` 的样本轴固定为 axis 0；`resample(..., axis=k)` 则显式指定输入组态轴，
  输出一律把 jackknife/bootstrap 重采样轴放在 axis 0，其余轴保留原顺序。例如
  `(2,3,4), axis=1 -> (Nsample,2,4)`。空样本轴报错，单样本 jackknife/SEM
  保留输出 shape 并以 `NaN` 表示统计误差不可得；整数输入升格到可表示
  `NaN` 的浮点 dtype。
- bootstrap 的默认 `seed=0` 保留历史可复现序列；显式
  `numpy.random.Generator` 由调用者持有并按原状态消耗，不得同时显式传 `seed`。
  `chunk_size` 只用于 bootstrap，不改变同 seed 的索引序列，并将当前索引临时量
  从 `O(Nsample*Nconf)` 降为 `O(chunk_size*Nconf)`；它不消除最终输出数组的
  `O(Nsample*...)` 内存。
- `resample` 对 NumPy/CuPy/torch 输入保留后端、设备和浮点/复数 dtype；`sem` 的复数
  输出 dtype 按前文映射为 `complex64 -> float32`、`complex128 -> float64`。SEM 的
  标准差必须在后端间使用同一 `ddof/correction` 定义，不得依赖 Torch 与 NumPy 不同的
  库默认值。
- `cov_mat` 只接受 `(Nsample, Nfeature)` 二维数组，空 feature 列报错；不得靠转置、
  补零或广播猜测样本轴。
- `fit` 要求 `y_coor=(Nsample,Ndata)`、有限的 `x_coor`，且基准模型输出、每次模型输出、
  参数和最终 residual 都有限并与 `Ndata` 对齐；`calc_chi2` 遇到 `NaN/Inf` residual
  必须报错。有限性失败不能通过 debug 的 NaN 填充伪装成成功拟合。
- 通用 `fit` 明确拒绝原生 complex `y_coor`。若观测量为 complex，调用者必须将同一批
  样本显式堆叠为 `[real, imag]` 的实数组，并使用与该堆叠完全对应的协方差；不能让
  隐式 cast 静默丢掉虚部。Hermitian complex covariance 只在明确使用支持它的
  `calc_chi2`/统计路径时成立，不能把两种约定混写。
- 无效输入应显式报错或按上层约定 mask；不能用零填充把缺测点变成物理数据。

## 协方差秩与模型 Jacobian

`FitParams.svdcut="auto"` 在进入 `gvar/lsqfit` 前解析为正 `1e-12`；显式
`svdcut=None` 严格关闭 SVD 调节，并要求 covariance 严格正定（所有特征值都为正）。若
可辨识门已通过但 covariance 仍有零模或非正模，必须在调用 `lsqfit` 前报错，不能暗中
恢复默认 cut。

`covariance_sample_rank` 在未调节相关矩阵上计算原始样本信息秩；
`covariance_effective_rank` 反映当前 `gvar.regulate`/`lsqfit` 的数据自由度。
`svdcut=None` 时二者相同；正 cut 抬升小相关本征值而不删模，负 cut 才删除小模，且
特征值恰好等于负 cut 边界时按 gvar 语义保留（`eigval >= cutoff`）。调节后
的 `effective_rank` 不能替代 `sample_rank`；bootstrap 扩增、协方差对角化和重复抽样也
不能制造原始组态没有的信息。

通用无-prior `FitParams`/`lsqfit` 拟合必须同时满足
`Ndata > Nparam`、`effective_rank > Nparam` 和 `sample_rank >= Nparam`。使用 prior 时，
`prior` 与 `p0` 必须有完全相同的参数键；prior 可提供参数约束，但零数据自由度或完全
没有样本涨落仍不能放行。该正自由度门是适配器报告契约，不是普遍的可辨识定理；专用
`fit_dispersion` 的满秩三点点估计可有 `dof=0`，必须显式标记没有 GOF。

协方差秩门通过后，在拟合参数附近计算数值模型 Jacobian：先按 covariance 的同一 SVD
模式白化，再按列缩放后求秩。`model_rank < Nparam` 即使 covariance 满秩，也表示模型
参数退化；该样本不写有限参数或 `chi2`，保留 `NaN`，不能依赖初值给出有限结果。近简并
场景应优先检查白化后的列秩和尺度条件数，不以“优化器收敛”代替模型可辨识性。

`cond=inf` 或 covariance 奇异本身不自动判失败；正 SVD 调节可以稳定数据自由度，但仍
须由独立的 `sample_rank` 门确认原始样本信息足以约束参数。协方差舍入修复只能把数值负
小模投影到零；已知精确核/零模必须保持为零。

### 模型 Jacobian

`FitParams.jacobian` 接受 `jacobian(x, p)`，返回两种等价形式之一：按参数名完整匹配的
mapping `{name: (Ndata,)}`，或列顺序严格等于 `list(fitpa.p0.keys())` 的
`(Ndata, Nparam)` 矩阵。每列必须有限；shape/key 不符必须报错。该 Jacobian 是拟合后
可辨识性检查所用的实参数方向，不应把 complex 参数或未声明的列混入；当前适配器不会
把它作为 `lsqfit` 优化器的显式导数接口，不能据此声称优化器收敛性能已改善。

没有解析 Jacobian 时，黑盒模型会在多尺度实数扰动上检查左右差分与中心差分。某列没有
有限、稳定且可分辨的响应时，必须报 `numerically indeterminate`；不能把“没有观测到响应”
静默当成零列或可辨识方向。真正平滑驻点只有在多尺度残差呈现可识别的正幂次趋势时才可
记为局部零导数；近简并仍须由白化、列缩放后的 rank 判定。

内建模型应优先提供闭式 Jacobian，当前入口包括：

| 模型 | 闭式实参数列 |
|---|---|
| ratio `R=c0+c1(e^{-dE t1}+e^{-dE t2})` | `∂c0=1`，`∂c1=e^{-dE t1}+e^{-dE t2}`，`∂dE=-c1(t1e^{-dE t1}+t2e^{-dE t2})` |
| energy `c0 e^{-E0 t}(1+c1e^{-dE t})` | `∂c0=e^{-E0t}(1+c1e^{-dEt})`，`∂c1=c0e^{-(E0+dE)t}`，`∂E0=-t C`，`∂dE=-t c0c1e^{-(E0+dE)t}` |
| FH `c0` | `∂c0=1` |

对应实现为 `model_ratio_jacobian`、`energy_model_jacobian` 和 `fh_model_jacobian`；
它们仍须经过同一 covariance 白化、列缩放和 rank 门。

## 非线性多态拟合的两层状态

对于两态及以上的非线性模型，先通过有限性/形状守卫，再把结果写成独立的状态层，不能
用一个“fit passed”字段覆盖不同问题：

1. `data_identifiability` 只表示数据是否在当前参数化下提供了所需的局部约束。无 prior
   时，它由已声明的 `sample_rank`、`effective_rank`、白化且列缩放后的 `model_rank` 和
   所需秩门决定；全部通过才可写 `True`/`identifiable`，任一失败写
   `False`/`statistically_unidentifiable`（产品层可按 finite mask 写
   `partially_identifiable`）。使用 prior 时写 `None`/`prior_constrained`，因为这是先验
   约束而非数据本身的可辨识证据。
2. `fit_quality_status` 独立记录拟合质量，例如 `chi2/dof`、Q、残差、pull、posterior
   与 prior 的收缩、收敛/边界信息以及各样本的 finite/usable 比例。缺少这些证据不能把
   它默认为“好”。
3. `physical_result_status` 独立判断目标物理主张是否获得支持：至少要结合谱式、边界和
   约束处理、能级/重叠参数、窗口与重采样稳定性，以及拟合质量。`data_identifiability=True`
   不得自动升级为 `fit_quality_status` 或 `physical_result_status` 的通过/可靠；一个边际
   参数稳定也不能替代其余态参数和模型假设的证据。证据不足时保留 `not_assessed` 或
   `unverified`。

### 条件数、有限率和稳定性元数据

`condition_number` 不是可独立解释的标量。每次报告它时，至少绑定：

- 对应对象是 covariance/correlation、未调节或已调节 covariance、白化且列缩放后的
  model Jacobian/design，还是其他矩阵；
- 矩阵形状、行/列或参数/特征顺序、范数/列缩放方式、白化方式和 SVD cut 状态；
- 若同时报告多个条件数，每个矩阵分别记录，不能只留下一个未命名的 `cond`。

对每个拟合参数和每个拟合诊断，在每个拟合窗口、模型变体和重采样方案下记录总样本数、
finite/usable 数和比例，以及中心值/误差和稳定性摘要（例如相对选定窗口或方案的漂移、
覆盖关系或分布摘要）。还要保存重采样方法、seed/索引、次数和 resample size，使“稳定”
可复核。这里不规定通用的 `cond`、AICc、Fisher、profile 或稳定性数值阈值；若采用项目
特定硬门，必须把门槛、适用对象和判定结果作为额外元数据写明。

### 哪些是硬门

本 reference 的硬门只约束数据可辨识性和输入安全：有限性/形状、声明的样本/有效/模型
秩门，以及 prior 状态门。AICc、Fisher 信息、profile likelihood 和重采样稳定性本身
不是本 reference 规定的通用硬门；它们用于模型比较、局部曲率/退化、非二次似然和结果
稳健性的诊断。它们的缺失或不稳定可以使 `fit_quality_status` 或
`physical_result_status` 保持未评估/未验证，但不能把 `data_identifiability` 偷换成质量
结论。

### FH 全逆窗口与能谱 AICc 的产品契约

`fit_constant_window(c0_zt)` 是一个专用闭式入口，输入轴固定为
`(n_tsep,n_sample)`。它直接使用未调节的普通全协方差逆，没有 `svdcut` 或伪逆路径；
因此这里必须满足 `sample_rank == effective_rank == n_tsep`。任一秩不足、逆不可得或
结果非有限时，返回与样本轴同长且全为 NaN 的 `c0_samples`，并把
`c0/c0_std/chi2/chi2_nocov` 设为 NaN，状态写
`statistically_unidentifiable`，同时保存 `fit_reason`、`n_data`、`n_sample`、
`sample_rank` 与 `effective_rank`。这条严格全秩门只属于该“普通全逆”入口；不能覆盖
上文允许显式 SVD 调节的通用 `fit`。

`fh_adaptive_windows` 只对显式 `identifiable`/`prior_constrained` 且有限的 `chi2`
执行窗口移动。当前窗不可辨识时必须原位停止该 `z` 的 chi2 驱动扫描，并把相同的
`fit_status/fit_reason` 传播到记录；不能继续搜索一个看似可画的窗。FH 参数图、窗口比较
和 best-fit 色带同样只消费显式可用状态与完整有限样本，缺少状态或只有 NaN 时跳过，
不得画零误差、虚构中心值或有限拟合带。

质子能谱 `model_selection="aicc"` 比较一态
`c0*exp(-E0*t)` 与两态 `c0*exp(-E0*t)*(1+c1*exp(-dE*t))`。若 AICc 选中一态，
NPZ/返回值必须写 `selected_model="one_state"`：保留通过一态状态门的
`c0/E0/chi2`，把固定 schema 中不属于该模型的 `c1/dE` 全部设为 NaN，分别记录
`ground_state_status` 与 `excited_state_status="practically_unidentifiable"`，并保存
逐参数状态和候选 AICc。有效质量图只能在选中模型状态显式可用且 `E0/chi2` 有限时画
E0 色带；否则显示不可用原因而不造带。AICc 偏好本身仍不能把
`fit_quality_status` 或 `physical_result_status` 升级为通过。

## prior 与状态字段

- prior 与 `p0` 键不一致时立即报错；prior 使拟合运行时，数据状态只能为
  `prior_constrained`，并写 `data_identifiability=None`/`pyqcd_data_identifiable=None`，
  这不是数据本身可辨识的证据。
- 无 prior 且白化、列缩放后的 Jacobian 及其他声明秩门足够时，才能写
  `identifiable`/`data_identifiability=True`；这仍不代表拟合质量或物理结果通过。
- 通用 `fit` 返回的 `last_fit` 只代表最后一个成功样本；产品层必须结合所有参数和
  `chi2` 样本的 finite mask 传播状态：全样本通过为 `identifiable`，部分通过为
  `partially_identifiable`，全失败为 `statistically_unidentifiable` 并保留 NaN。任何
  `prior_constrained` 都必须保持数据可辨识字段为 `None`。
- ratio、energy、disconnected 和 TMD ratio 在秩不足时跳过 `lsqfit`；参数与 `chi2` 写
  `NaN`，状态为 `statistically_unidentifiable`，并记录 `effective_rank`、`sample_rank`
  和 `required_rank`。单组态不可辨识，不得用 `0` 冒充矩阵元或有限拟合结果。
- TMD 的三参数拟合状态与 `c0` plateau 状态分别落盘；可用 plateau 不能覆盖或升级不可
  辨识的拟合状态。低 Q、大 `chi2/dof`、先验主导和窗口不稳定是通过秩门后的独立诊断。

统计结果至少保留 `Ncfg`、方法、seed、covariance 约定、条件数及其对应矩阵、SVD cut、
原始样本秩、调节后有效秩、所需参数秩、窗口、模型变体、自由度、`chi2/dof`、Q 值、各
参数/诊断的 finite/usable 计数与比例、窗口/重采样稳定性摘要，以及三个独立状态字段；
这组元数据用于区分“数据可辨识”“拟合质量”和“物理主张得到支持”。

## 可执行 contract

以下入口是当前仓库中可直接运行的统计边界与可辨识性 contract；它们验证实现契约，不能
替代真实系综、完整物理链或系统误差账本：

```bash
# 样本轴、有限性、协方差和单组态状态
python -m pyqcd.testing._statistics_edge_contract

# 任意输入样本轴、分块 bootstrap、RNG 和后端/dtype 契约
python -m pyqcd.testing._bootstrap_resampling_contract

# 相关拟合、prior、样本/有效秩和模型 Jacobian
python -m pyqcd.testing._fit_identifiability_contract

# 上层 ratio/energy/FH/TMD 状态传播
python -m pyqcd.testing._fit_status_propagation_contract

# FH 普通全逆窗口、稀疏 t_sep 滑窗与不可辨识传播
python -m pyqcd.testing._fh_window_contract

# 质子能谱一态/两态 AICc 选择与参数状态
python -m pyqcd.testing._energy_model_selection_contract

# 专用色散与连续外推的可辨识性边界
python -m pyqcd.testing._dispersion_identifiability_contract
python -m pyqcd.testing._extrapolate_identifiability_contract
```

若上述 contract 或上层运行门失败，保留 raw/intermediate 和失败状态；不要通过默认值、
先验或可用图表升级为真实物理结论。
