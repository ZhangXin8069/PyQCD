# PyQCD C3/C2 -> 裸矩阵元 c0 统计与拟合审计

审计日期：2026-10-22（Asia/Shanghai）  
代码快照：`HEAD=3e1edd4`，含审计期间工作树改动；目标文件 SHA-256 见文末。  
审计范围：`_tmd_ratio.py`、`_disconnected.py`、`_fitter.py`、`pipeline/_tmd9.py` 中的 OPE/缓存/自归一、donghx 参考 ratio/fit 代码，以及现有 TMD 产物。  
写入约束：本任务只创建本报告；未修改生产代码、`refer/`、Git 索引或系统配置，未运行会创建临时目录的测试套件。

## 结论

当前链路不能作为“胶子 TMD-PDF 物理结果”的证据。最严重的问题不是统计误差大小，而是 C3 因子化所需的 C2 源时间轴在生产缓存中已经丢失，分析层又对该源平均量做了第二次源平移；随后真空估计量与上游参考定义不一致。当前真实 conf6250 缓存中的非零 ratio 可由这次错误平移完整复现，而按缓存实际语义去掉二次平移后，拟合窗 ratio 为数值零。

同时，代码把“局部秩通过”保存成通用 `fit_status=identifiable`，把“plateau 窗口内有非零样本涨落”保存成 `plateau_status=identifiable`，却没有拟合质量或物理支持状态。下游据此生成 c0 图和 matched TMD-PDF 文件。历史 `initial_run`、`extended_run`、`final_run` 的 P200/P400 拟合报告逐字节相同，不能视为三个独立验证；这些报告的 TMD c0/PDF 数值应撤回为不可靠/仅历史诊断。

## 严重级别摘要

| ID | 级别 | 问题 | 结论 |
|---|---|---|---|
| C1 | Critical | 生产 C2 只有源平均 `(Nt,)`，分析层又把它当绝对时间序列滚动 | 当前 C3 的源轴和关联结构错误 |
| C2 | Critical | 非 TMD `C3_disc` 与 `configuration_ratio` 对因子化 C3 恒等为零 | 两条路径不能产生物理 c0 |
| C3 | Critical | 默认 ensemble 真空扣除先逐组态减 source mean，参考实现则做全局 covariance | 小 ensemble 下估计量可相差近 100% |
| H1 | High | rank 状态、plateau 状态和物理结果状态未分层，失败仍出图/出 PDF | 状态升级，物理结论无效 |
| H2 | High | `c0/c1/dE` 在近简并区实际不可辨识，硬编码 `svdcut=1e-6` 无敏感性证据 | 秩通过不能排除大 chi2/不可用参数 |
| H3 | High | z=0 同动量 shape normalization 被当作 PDF 自重整化输入 | 缺少 `Pz=0` 分母、SNR 和稳定性门 |
| H4 | High | 历史 final report 是 initial report 的逐字节拷贝，含大量病态拟合参数 | 历史 TMD 数值不可复现、不可作物理证据 |
| M1 | Medium | 中心值取重采样拟合均值，报告却打印最后一个 replica 拟合 | 中心估计量定义不清 |
| M2 | Medium | bootstrap 共享索引靠两个默认 seed 隐式相同，索引/seed 未落盘 | 可复现性和共享样本轴无审计凭证 |
| M3 | Medium | OPE 缓存未绑定输入 gauge 内容或实现源码哈希 | 规范场替换后可静默复用 stale OPE |

## Critical

### C1. C2 源轴丢失并发生二次时间平移

**位置**

- `pyqcd/pipeline/_steps.py:1034-1059`：2pt 对 `t_src` 循环后写入 `acc[dt] += val / NT`，最终缓存只有源平均 `(Nt,)`。
- `pyqcd/pipeline/_tmd9.py:509-553`：strict 2pt cache contract 的 shape 固定为 `(Nt,)`。
- `pyqcd/analysis/_tmd_ratio.py:203-213`：把上述 `(Nt,)` 交给 `_relative_correlator`，后者把它解释为绝对时间序列并再次 `roll(-source)`。
- `pyqcd/analysis/_disconnected.py:24-40`：`_relative_correlator` 的平移在“输入确实是绝对时间序列”时是正确的；错误在于生产输入已经是相对、源平均 C2。
- 参考 `refer/donghx/_Gradient_Flow_GluonPDF/calc_ratio_matched.py:201-216`：保留完整 `(tsep,t_source)` C2，数据源为 `(Nt,Nt)`。
- 参考 `refer/donghx/_Gradient_Flow_GluonPDF/calc_ratio_matched.py:406-416`：对每个 source 用 `(source+insertion)%Nt` 取 OPE 时间片。

**真实缓存量化**

只读使用 `pyqcd/testing/tmd/artifacts/work/data/conf6250` 的 C2 和 OPE：

```text
C2 经错误源平移后的 source-axis max(ptp)/max|C2| = 0.294693/0.275308
当前错误 ratio 在拟合窗的 RMS                  = 3.38715
按“已源平均相对 C2 不再平移”的 ratio RMS      = 2.50331e-15
```

后者的零不表示物理 c0 必为零，而是说明现有 1D C2 没有足够信息构造 C3；当前非零结果则明确来自二次平移制造的源依赖。

**复现**

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
from pathlib import Path
import h5py, numpy as np
from pyqcd.analysis._disconnected import _relative_correlator
from pyqcd.analysis._tmd_ratio import _disconnected_ratio
root=Path('pyqcd/testing/tmd/artifacts/work/data/conf6250')
with h5py.File(next(root.glob('corr_pp_*.h5')),'r') as f: c2=f['data'][...]
with h5py.File(next(root.glob('tmd_ope_*.h5')),'r') as f: o=f['data'][...]
nt,dtmax,nz,nb=72,20,13,5
ot=o.transpose(2,0,1)
orel=np.zeros((1,nt,dtmax,nz,nb))
for s in range(nt): orel[0,s]=np.roll(ot,-s,axis=0)[:dtmax]
mask=np.zeros((dtmax,dtmax),bool)
for dt in range(7,11):
    for tau in range(3,dt-2+1): mask[dt,tau]=True
for label,c2rel in [
    ('double_shift',_relative_correlator(c2[None,:],dtmax)),
    ('no_shift',np.broadcast_to(c2[None,None,:dtmax],(1,nt,dtmax)).copy()),
]:
    c3=np.zeros((1,nt,dtmax,dtmax,nz,nb))
    for dt in range(dtmax):
        for tau in range(dt+1):
            c3[0,:,dt,tau]=orel[0,:,tau]*c2rel[0,:,dt,None,None]
    r=_disconnected_ratio(c2rel,orel,c3,estimator='ensemble')
    print(label, float(np.sqrt(np.mean(r[0][mask]**2))))
PY
```

**影响**

当前 `ensemble` ratio、其 bootstrap/jackknife 协方差、c0 拟合和 plateau 均建立在错误源轴之上。即使后续拟合数学上收敛，目标矩阵元仍不是上游定义的 disconnected C3/C2。

**最小修复**

- 将 2pt 产物和 strict cache 改为保留 `C2[t_sink,t_source]`，或至少保存 source-dependent `C2[source,dt]`；不得只保存源平均 `(Nt,)`。
- TMD 分析接口拒绝 1D C2 作为物理 C3 输入，避免静默滚动。
- 若上游已提供相对时间 `C2[source,dt]`，直接索引，不再调用绝对时间 `_relative_correlator`。
- 对非极化实数通道，显式记录“已证明虚部可忽略”的投影；不能把通用复数 C2 入口默认转成实部。

**新增测试断言**

- producer 测试：对非对称 source/sink synthetic C2，缓存往返后 source 轴逐元素一致。
- shift 测试：输入相对 `C2[source,dt]` 时输出必须等于输入切片，不能再次平移。
- 与 upstream oracle 测试：两组态以上、source-dependent C2 上，C3、denominator、jackknife 样本逐项一致。
- 回归测试：若传 1D 源平均 C2 给物理 ratio 入口，必须抛出 `ValueError` 或返回 `insufficient_source_axis`。

### C2. `configuration_ratio` 和非 TMD disconnected 路径恒等为零

**位置**

- `pyqcd/analysis/_tmd_ratio.py:90-99`：`configuration_ratio` 使用 `(C3-C2*O)/C2`，但 `C3` 在 `_tmd_ratio.py:215-221` 被构造为 `C2*O`。
- `pyqcd/analysis/_disconnected.py:515-530`：`run_disconnected_ratio` 同样先构造 `corr3=OPE*C2`，随后 `corr3_disc=corr3-C2*OPE`，再除以 `C2+1e-30`。
- 参考 `calc_ratio_matched.py:308-328`：真空项是全局 ensemble mean，而不是用同一组态上的同一个 OPE 因子再次相乘。

**量化**

任取非零随机 C2、OPE，按当前构造：

```text
max|C3_disc| = 0.0
max|configuration ratio| = 0.0
```

因此 `configuration_ratio` 不是“历史估计量的小差异”，而是当前因子化构造下的代数零；非 TMD 模块也完全相同。`eps=1e-30` 还会把接近零的 C2 分母变成静默零或爆炸，而不是失败关闭。

**复现**

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.analysis._tmd_ratio import _disconnected_ratio
rng=np.random.default_rng(8)
ns,nsrc,ndt,nz,nb=4,6,4,2,1
c2=rng.uniform(.5,1.5,(ns,nsrc,ndt))
op=rng.normal(size=(ns,nsrc,ndt,nz,nb))
c3=op[:,:,None,:,:,:]*c2[:,:,:,None,None,None]
for est in ('ensemble','configuration_ratio'):
    r=_disconnected_ratio(c2,op,c3,estimator=est)
    print(est,float(np.max(np.abs(r))))
PY
```

**影响**

非 TMD `run_disconnected_ratio` 不能产生非零物理 ratio；任何由全零 ratio 得到的拟合参数和 chi2 都没有物理含义。`configuration_ratio` 选项也不应再用于历史复现声明。

**最小修复**

- 删除或 fail-closed 禁用 `configuration_ratio`，状态写 `factorized_configuration_ratio_degenerate`。
- 非 TMD `run_disconnected_ratio` 改用与 reference 一致的 ensemble vacuum subtraction；若只能得到 factorized C3，应只保存 covariance 诊断，不得进入三参数 c0 拟合。
- 移除 `C2+1e-30`；改用相对尺度阈值并保存 masked/失败通道。

**新增测试断言**

- factorized C3 + `configuration_ratio` 必须返回不可辨识状态，不得返回有限 c0/chi2。
- 非 TMD 模块对任何输入不得生成全零 ratio 后仍执行 `lsqfit`。
- C2 分母为零或低于声明相对阈值时，对应通道必须是 NaN/status，而不是被 `1e-30` 修补。

### C3. ensemble 真空扣除与上游全局 ensemble 定义不一致

**位置**

- 当前 `pyqcd/analysis/_tmd_ratio.py:72-88`：
  `mean_i[ mean_s(C_i,s O_i,s) - mean_s(C_i,s) mean_s(O_i,s) ] / mean(C2)`。
- 参考 `calc_ratio_matched.py:308-324`：
  `mean_i,s(C_i,s O_i,s) - mean_i,s(C_i,s) mean_i,s(O_i,s)`，再做 denominator division。

两者之差严格为 source-average 量的 configuration covariance：

```text
current - reference = -Cov_i(mean_s C_i,s, mean_s O_i,s)
```

在 10 个组态的小 ensemble 中，这一项不是统计噪声可以忽略的高阶修正。

**量化**

8 个组态的确定性相关样本得到：

```text
current per-config/source subtraction = 0.0059633
upstream global ensemble covariance   = 0.297244
relative difference                   = -97.9938%
```

**复现**

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
rng=np.random.default_rng(410); nconf,nsrc=8,6
c=np.empty((nconf,nsrc)); o=np.empty((nconf,nsrc))
for i in range(nconf):
    ci=0.3*rng.normal(); oi=0.8*ci+0.15*rng.normal()
    c[i]=1.0+ci+0.15*rng.normal(size=nsrc)
    o[i]=2.0+oi+0.25*rng.normal(size=nsrc)
cb=np.mean(c,axis=1); ob=np.mean(o,axis=1)
current=np.mean(np.mean(c*o,axis=1)-cb*ob)
upstream=np.mean(c*o)-np.mean(c)*np.mean(o)
print(current,upstream,100*(current-upstream)/abs(upstream))
PY
```

**影响**

估计量定义会改变 c0 中心值。若继续使用与上游比较的数字，必须统一为同一个 ensemble 定义；否则代码/报告之间的百分比差异可由估计量而非数据造成。

**最小修复**

- 默认采用 reference 的全局 ensemble covariance，并把公式写入 artifact metadata。
- 若要保留逐组态 source covariance 估计量，必须另命名、显式选择、证明物理目标并提供与全局估计量的差异诊断。
- denominator 必须与该估计量使用同一组重采样样本，并在同一样本内重算。

**新增测试断言**

- 两组态层级相关 synthetic 上，PyQCD 输出与闭式全局公式逐项 `allclose`。
- 同一输入下记录两种估计量差值，若差值超过声明阈值必须标为估计量敏感。

## High

### H1. 数据可辨识性、拟合质量和物理状态被合并成一个字段

**位置**

- `pyqcd/analysis/_fitter.py:865-893`：`pyqcd_fit_status` 只由 covariance/effective rank 和模型 Jacobian rank 决定，随后直接成为 `identifiable`。
- `pyqcd/analysis/_tmd_ratio.py:377-397`：把该状态保存为通用 `fit_status`；没有 Q、chi2 门、参数稳定性、窗口稳定性或物理状态。
- `pyqcd/analysis/_tmd_ratio.py:117-145`：plateau 只检查样本有限且存在非零涨落，就写 `plateau_status=identifiable`，并把人工值 `sample_rank=1` 落盘。
- `pyqcd/analysis/_tmd_ratio.py:482-529`：`plot_tmd_c0` 不检查 `fit_status` 或 `plateau_status`，任何 NaN/病态 c0 都会进入图。
- `pyqcd/testing/tmd/pdf_workflow.py:197-212`：c0 图无条件生成，均值/误差无条件保存。
- `pyqcd/testing/tmd/pdf_workflow.py:221-279`：所谓 verified PDF 输入只要求 plateau 顶层状态为 `identifiable`。
- `pyqcd/testing/tmd/pdf_workflow.py:283-364`：使用该 c0 直接生成 quasi/matched TMD-PDF 和“matched”图，不传播 fit quality/physical status。

**合成量化**

当前 `fit` 在一个确定性 bootstrap 常值近似数据上给出：

```text
last_status=identifiable
Ndata=14, sample_rank=9, effective_rank=14
chi2/dof=39880.87
```

即模型/协方差局部秩通过，但拟合质量灾难性失败，仍被标成 `identifiable`。这不是说 rank 门实现错误；错误是产品层把 rank-only 状态当成可用拟合。

**Plateau 合成量化**

构造明显随 `dt,dtau` 增长的窗口，只有样本噪声：

```text
window point means range = [3.0658, 4.2055], spread=1.1397
plateau_status            = identifiable
```

说明当前状态没有检验“平台是否平坦”。

**真实失败产物**

`work` 运行只有一个组态，拟合报告正确写 `statistically_unidentifiable`，但仍有：

```text
c0_tmd_proton_P200.png
ratio_proton_P200.png
tmd_summary.json
```

`tmd_summary.json:1-50` 只描述 workflow/shape/time，没有失败状态。图形和 summary 仍可被外部读者当成完整运行产物。

**最小修复**

- 输出至少三个独立字段：`data_identifiability_status`、`fit_quality_status`、`physical_result_status`。
- plateau 增加独立的质量门：常数模型 chi2/Q、相邻窗口漂移、cut 敏感性、完整有限样本率和逐参数 spread。
- `plot_tmd_c0` 只在对应 z/b 的显式状态可用时绘制；失败/不可辨识点不画色带、零误差或有限中心线。
- `pdf_workflow` 只有在 `physical_result_status` 显式通过时才能写出名为 `matched_tmd_pdf_*` 的物理产物；否则文件名和 summary 均标 `diagnostic`，并返回非成功阶段状态。
- 单组态/低秩 summary 必须列出 skipped phases 和失败原因。

**新增测试断言**

- rank 通过但 `chi2/dof=1e4` 的 synthetic 必须为 `data_identifiable` 且 `fit_quality=failed`，不得 `physical_result=passed`。
- 明显非平坦窗口即使样本有涨落，`plateau_status` 也不能是物理可用的 `identifiable`。
- 全 NaN 或 unidentifiable fit 时，`plot_tmd_c0` 返回空图列表或只输出带失败水印的诊断图。
- `run_analysis` 的单组态运行必须在 summary 中写 `pdf_chain=skipped`，不能只留下正常 shape 字段。

### H2. `c0/c1/dE` 的 practical identifiability 未检查，SVD 容易制造“好 chi2”

**位置**

- `pyqcd/analysis/_disconnected.py:443-463`：模型 `R=c0+c1 exp(-dE tau)+c1 exp(-dE(dt-tau))`。
- 当 `dE -> 0`，`c0` 列和 `c1` 列严格共线，`c0` 与 `c1` 的拆分实际不可辨识。
- `pyqcd/analysis/_tmd_ratio.py:358-379` 硬编码 `svdcut=1e-6`，没有 `1e-12`/`1e-8`/`1e-6` 敏感性扫描。
- `_fitter.py:204-256` 正确区分 sample/effective rank，但 rank 通过不等于参数误差有限或物理可用。

历史 P200 `z=0,b=0` 报告给出 `c0=3203496`, `c1=-1.60176e6`, `dE≈-2e-6`。用当前模型在这些参数处计算 covariance-independent 的列缩放设计矩阵：

```text
column-scaled condition = 1.02034e12
smallest singular value  = 1.69486e-12
```

该方向远低于当前黑盒 Jacobian 的典型 rank 阈值，说明历史有限 c0 参数是近零 `dE` 盆地的病态伪解。当前代码会更容易拒绝这一具体点，但仍缺少 practical condition/parameter-uncertainty 门。

**影响**

- `chi2/dof` 很小可能来自正 SVD cut 抬升零模，而不是模型真正描述了数据。
- `c0` 的数值可随优化初值、SVD cut 和最后一个 replica 大幅跳变；不能据此导出自重整化分子。
- 当前 synthetic 已证明 rank 通过时 `chi2/dof` 可达 `4e4`；反向也会出现低 chi2 但参数近简并。

`pyqcd/analysis/_tmd_ratio.py:178-190` 还根据 `Nconf > Ndata_est` 自动在 jackknife 与 `Nsample=200` bootstrap 之间切换，并注释称 200 replicas 可得到“满秩协方差”。这不成立：bootstrap 只从 `Nconf` 个原始组态抽取，原始样本协方差秩至多约 `Nconf-1`。在 `Nconf=10,Ndata=14` 的 synthetic 中实际为 `sample_rank=9,effective_rank=14`；正 `svdcut` 抬升了 5 个原始零模。因而 `dof=11` 是调节后自由度，不是 11 个独立原始统计方向，且改变拟合窗可能改变重采样估计量。

**最小修复**

- 在 artifact 中记录白化、列缩放后的 model Jacobian SVD、条件数、最小参数误差和 `dE` 置信区间。
- 对 `dE` 接近零、`c1` 爆炸、`c0` 被 `c0/c1` 抵消的方向标 `practically_unidentifiable`。
- 固定窗口上做 `svdcut` 和初值敏感性；结果依赖任一选择时，physical status 保持 `unverified`。
- 若采用 prior，只允许 `prior_constrained`，不能升级为 data identifiable。

**新增测试断言**

- `dE=0` 且 `c1` 很大时，模型 Jacobian rank 为 1（或明确的 `<3`），返回 NaN/不可辨识。
- 同一数据在 `svdcut={1e-12,1e-8,1e-6}` 下若 c0 漂移超阈值，`fit_quality_status` 不得通过。
- `chi2/dof=1e4` 而 rank=3 的用例必须保留有限参数作为诊断，但状态不能是可交付物理结果。
- `Nconf=10,Ndata=14` 必须分别保存 `sample_rank<=9` 与 regulated effective rank；artifact 不得写“bootstrap 满秩”。
- 固定组态集合时，改变窗口不得静默改变 jackknife/bootstrap 方法；方法必须显式配置并记录。

### H3. z=0 同动量 shape normalization 被当成物理自重整化

**位置**

- 当前 `pyqcd/pipeline/_tmd9.py:907-929` 已明确注释：`c0(z,b)/c0(z=0,b)` 是 shape-normalization 诊断，不是零动量混合方案分母。
- `pyqcd/pipeline/_tmd9.py:1-16` 模块说明仍写 `hB(z,b,Pz)/hB(0,b,Pz=0)`，与实现不是同一量。
- 物理 ratio/hybrid 定义见 `pyqcd/renorm/_hybrid.py:4-7`：短距应除 `hB(z,Pz=0)`，不是同一 `Pz` 下的 `hB(z=0)`。
- `pyqcd/testing/tmd/pdf_workflow.py:265-279` 只拒绝精确零分母，随后接受均值有限的结果为 PDF 输入。
- `pyqcd/testing/tmd/pdf_workflow.py:355-360` 只保存 plateau 的 `input_status=identifiable`，没有保存 `shape_normalization_only`。

**合成量化**

10 个样本中令 z=0 样本均值为 `0.0055`、SEM 为 `0.02348`（`mean/SEM=0.234`），且所有样本均非零：

```text
status             = identifiable
hR(z=1) mean       = 7.0403
hR(z=1) SEM        = 5.4938
hR(z=1) sample range = [-11.5, 48.0]
```

代码会接受该输入为 finite/verified，尽管 z=0 分母与零完全相容。

**复现**

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.analysis._tmd_ratio import _plateau_status_metadata
from pyqcd.pipeline._tmd9 import self_renormalize
c0=np.empty((10,3,1))
c0[:,0,0]=[-.12,-.08,-.04,-.02,.005,.01,.03,.06,.09,.12]
c0[:,1,0]=.20+.01*np.arange(10); c0[:,2,0]=.10+.008*np.arange(10)
meta=_plateau_status_metadata(np.zeros((10,20,20,3,1)),c0,20,7,10,6)
hR,_=self_renormalize(c0); z0=c0[:,0,0]
print(z0.mean(),z0.std(ddof=1)/np.sqrt(10),meta['plateau_status'])
print(hR[:,1,0].mean(),hR[:,1,0].std(ddof=1)/np.sqrt(10),hR[:,1,0].min(),hR[:,1,0].max())
PY
```

**影响**

即使 `matching_status.json` 当前可能写 `physical_acceptance=false`，生成的 `matched_tmd_pdf_xg.npy` 和图中的 “NLO matched” 标题仍没有自动传播这一限制。更根本地，同 `Pz` 的 z=0 shape normalization 不执行题目声明的零动量自重整化。

**最小修复**

- 将函数改名为 `shape_normalize_z0`，artifact 明写 `normalization_kind="same_Pz_z0_shape_only"`。
- 物理 PDF 入口要求 `c0_pz0` 形状相同的零动量矩阵元，并调用 `tmd_renormalize_hybrid`；没有 `Pz=0` 数据时只输出 diagnostic。
- 增加 z=0 分母支持门：同号、相对误差阈值、留一组态稳定性、jackknife/bootstrap 符号翻转统计。
- `pdf_chain_input_status` 保存 normalization kind、denominator SNR、完整状态和 `physical_acceptance`。

**新增测试断言**

- z=0 分母跨零或 `mean/SEM < declared_threshold` 时，PDF 输入必须被拒绝或降级。
- 只有 `Pz>0`、无 `Pz=0` 时，不得写出 `matched_tmd_pdf_xg.npy`；应写 `diagnostic_shape_normalized_*`。
- `self_renormalize` 的输出在 `z=0` 必须精确为 1，同时状态仍为 `shape_only`。

### H4. 历史 final report 不可作为独立验证，TMD 数值应撤回

**证据**

```text
initial_run  P200/P400 report SHA-256
final_run    P200/P400 report SHA-256 完全相同
extended_run P200/P400 report SHA-256 也相同
```

命令：

```bash
sha256sum pyqcd/testing/tmd/artifacts/{initial_run,extended_run,final_run}/analysis/tmd_ratio/1_fit_report_P200.txt
sha256sum pyqcd/testing/tmd/artifacts/{initial_run,extended_run,final_run}/analysis/tmd_ratio/1_fit_report_P400.txt
```

三个 run 中每个都有 10 个 TMD 报告；每个 run 的 650 个 `(z,b)` block 全部为 `cond=inf`：

```text
initial_run  files=10 blocks=650 cond_inf=650
extended_run files=10 blocks=650 cond_inf=650
final_run    files=10 blocks=650 cond_inf=650
```

历史报告缺少当前要求的 `fit_status`、sample/effective rank、model rank、Q/quality 状态和 bootstrap 索引。`initial_run/.../P200.txt:11-20` 给出病态参数 `c0=3203496`, `c1=-1.60176e6`, `dE≈-2e-6`；`P400.txt:11-20` 给出 `c1=-4e7`, `dE=5±2508`。这些不是可交付的中心参数。

报告头部还写 `Nsample=10, jackknife=True`，但当前默认窗口 `dt=[7,10],cut=6` 有 14 个数据点，当前逻辑会选择 `bootstrap,Nsample=200`。因此这些报告不仅缺状态，也不是当前实现的产物。

历史文档 `final_run/docs/analy_test9_2_20260820.tex:163` 声称已完成梯度流重整化全链，`:236` 声称 z=6 plateau 与 c0 一致，`:264` 声称满足闭环交付。对照同 run 的报告，`z=6,b=0` 的最后 replica 参数为 `c0=11±11`, `c1=-1.97e14`, `dE=10.23±0.60`；没有中心参数数组可证明文档所称的 0.02--0.1 c0。该物理结论不应继续引用。

**最小修复**

- 将 `initial_run`、`extended_run`、`final_run` 的 TMD c0/PDF 数值标为 `invalidated_by_statistics_audit`，保留原始文件但禁止作为物理证据。
- 新 run 必须保存 `conf_ids`、exact estimator、seed/indices、ranks、quality/physical status、z=0 normalization policy 和 cache input hashes。
- 只有在修复 C1--C3 后重算 10 组态并完成真实数据验证，才能重新生成物理报告。

## Medium

### M1. 没有 full-sample central fit；报告打印最后一个 replica

`pyqcd/analysis/_fitter.py:839-896` 对每个重采样样本逐一 `lsqfit`，只返回最后一个成功样本 `last_fit_info`。`pyqcd/analysis/_tmd_ratio.py:403-413` 又把该对象格式化进报告。

因此：

- 报告中的 `Parameters` 是最后一个 replica 的结果，不是 full-sample central fit。
- c0 均值是 bootstrap replica 均值或 jackknife mean；jackknife 没有保存 full central fit，也没有 bias-corrected jackknife 中心。
- `chi2_plot` 是 replica chi2 均值，不能替代 full-data GOF/Q。

参考 `calc_ratio_matched.py:373-382` 至少同时返回 jackknife mean 和 bias-corrected ratio，便于明确中心定义。

**最小修复**

- 先对原始 sample-average data 与同一 covariance 做 full central fit，保存中心参数/Q/chi2。
- bootstrap 只产生误差；jackknife 可选 bias-corrected 中心；报告不得把任意 replica 标成 central。

**测试断言**

- 报告必须含 `central_fit` 和 `replica_summary` 两段。
- `last_fit_info` 不得被序列化为 central parameters。
- jackknife artifact 同时保存 full、mean_jack、bias_corrected 三个定义及公式。

### M2. 共享重采样索引是隐式约定，未保存

`pyqcd/analysis/_tmd_ratio.py:227-229` 对 C2、OPE、C3 分别调用 `resample`。当前 bootstrap 恰好共享索引，因为 `_disconnected.py:175-177,213-214` 每次默认 `seed=0` 并新建同序列 RNG；jackknife 也天然共享 delete-one 索引。

内存检查得到共享索引代理残差 `8.9e-16`，所以当前数值路径确实共享样本。但：

- artifact 只保存 `resampling_kind` 和 `Nsample`，没有 seed、indices 或 index hash。
- 一次 API 默认值改变或未来支持自定义 `rng`，三个量可能悄然错配。
- `0_fit_data_*.npz` 不保存 `conf_ids`，无法审计哪一组态进入协方差。

**最小修复**

- 一次生成 `indices(Nsample,Nconf)`，显式索引 C2、OPE、C3；保存 indices、seed、index SHA-256、`conf_ids`。
- jackknife 保存被删除的 sample id 序列。

**测试断言**

- 对任意 Nconf/Nsample/seed，三数组的每个 bootstrap replica 使用逐元素相同 index row。
- artifact 的 index hash 可由 `conf_ids + seed + method` 重算。

### M3. OPE/flow cache 未绑定输入 gauge 和实现源码

`pyqcd/pipeline/_tmd9.py:374-388` 的 flow contract 只有 conf id、tau、eps、precision、lattice/shape；`682-705` 的 OPE contract 只有 flow controls、z/b/direction、staple/color normalization。缓存 HDF5 attrs 的实际 key 也没有 gauge file hash、ensemble id、git commit 或运算符源码 hash。

`_OPE_ALGORITHM_VERSION="...v1"` 是手工版本串。只要实现或输入 gauge 内容改变但版本串未改，旧 cache 仍会被接受。

**影响**

同一 `conf_id` 的 gauge 文件被替换、系综/涂抹方案混用，或 OPE 代码未 bump version 时，可能静默使用 stale 数据。缺失 cache 输入绑定时，物理重现性不能只靠当前 contract 证明。

**最小修复**

- contract 增加 input gauge SHA-256、ensemble identifier、operator/staple source revision 和 flow implementation hash。
- cache hit 时把 provenance 写进 run manifest。

**测试断言**

- 同 conf id、不同 gauge bytes 必须 cache miss。
- 同参数、bump implementation revision 必须 cache miss。

## 已核对且未发现同等级错误的统计基元

- `sem(..., jackknife=True)` 与 delete-one 公式一致：对 sample `[1,2,4,8,9]`，实现值和 `sqrt((N-1)/N * sum((theta_i-mean)^2))` 完全相等（差 0）。
- bootstrap `sem` 使用 population std，不再除 `sqrt(Nsample)`。这与本仓库声明的 API 一致；参考 `fit_bare_latest.py:254` 和 `fit_cpu.py:49,206` 使用 `ddof=1`，因此跨代码对比时 PyQCD 在 `B=200` 会低 `sqrt(199/200)=0.997496`，即约 0.25%。这是定义差异，不是当前主要物理误差。
- `cov_mat` 的 jackknife 因子 `(n-1)/n sum` 和 bootstrap 因子 `1/n sum` 与 `sem` 一致。
- `_fitter.py` 对 sample rank 与 SVD-effective rank 的区分本身符合统计契约；问题在 product 状态分层，不在这两个诊断函数的定义。
- `resample` 的单组态 jackknife 返回 NaN；TMD fit 分支也在一组态时跳过 `lsqfit`，这部分 fail-closed 正确。
- cache JSON/SHA 自洽校验和原子写入实现正确；缺陷是 provenance 字段不足。

## 最小修复顺序

1. 立即停止把当前 TMD c0/PDF 当物理结果：把历史 run 标 invalidated，并让 PDF chain 在 source-specific C2 不存在时 fail-closed。
2. 重做 2pt cache contract，保存 source-sink C2，修正 source shift，选用并冻结 ensemble vacuum 定义。
3. 把 fit/plateau 的三层状态和 z=0 normalization policy 写入 artifact；图、PDF、summary 全部消费这些状态。
4. 增加 practical identifiability、SVD/初值/window 敏感性和 full-sample central fit。
5. 保存共享 resampling indices/seed、cache 输入哈希及实现版本，再重算 10 组态。

## 阻塞与限制

- `pyqcd/testing/tmd/artifacts` 中没有 initial/final 10 组态对应的原始 OPE、C2、ratio 数组，只有报告/图；无法独立逐位复算历史 c0。现有真实二进制 cache 仅覆盖单组态 conf6250，因此只能验证输入语义和估计量错误，不能替代 10 组态物理门。
- 审计期间工作树中的 `pipeline/_tmd9.py` 和 `testing/tmd/pdf_workflow.py` 被并行修改；本报告行号和结论对应下列快照哈希。审计没有修改这些文件。

## 审计快照 SHA-256

```text
e6cf480ae1a5cc4d1b4324d5e0e6363f0ef6c47f5a405db86feca03c3c3e0796  pyqcd/analysis/_tmd_ratio.py
90be7d124d9816d0e7285261356cf882aa625709fb5dc77816bcd13c4c3b7be2  pyqcd/analysis/_disconnected.py
6fe47425b84bc5e25b52c694a1dcfa9b53cf3125578af6351b1577263169b960  pyqcd/analysis/_fitter.py
b2db6db24ba8c13e7a16bc14ea97ee6ca42d4b38c35deb3c965b260bfdcd3938  pyqcd/pipeline/_tmd9.py
62ce5abacf0815a96aa2928f50169fb8b36e071a33e8b8a5becf6880574437f3  pyqcd/pipeline/_steps.py
b81104ded51a73df8b897ba0772019c123b41ccd1b373dcad4fbb1b65b939bac  pyqcd/testing/tmd/pdf_workflow.py
1ff5e2c6cacea9b595b9dbf6b79e7637dd08ca70a20e95922d84b1a48d5adc7b  refer/donghx/_Gradient_Flow_GluonPDF/calc_ratio_matched.py
029dc99810e8f064d6e567a223887a2f4a33760232578259a3ecc0ae053c80ab  refer/donghx/_Gradient_Flow_GluonPDF/fit_bare_latest.py
7482f377edfbdc7f14dfae8094bde892ba661558cea225d042b2e5d91e2bbfe9  refer/donghx/_Gradient_Flow_GluonPDF/fit_cpu.py
```
