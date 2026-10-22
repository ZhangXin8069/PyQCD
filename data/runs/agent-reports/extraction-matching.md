# 胶子 TMD-PDF 提取与匹配审计

审计日期：2026-10-22（Asia/Shanghai）  
审计范围：重整化矩阵元 \(\to\) 准 TMD-PDF \(\to\) CS 核 \(\to\) NLO 匹配 \(\to\) “最终 PDF”状态链。  
审计方式：只读源码、参考实现、既有 cache 与合成极限；未修改生产代码、`refer/` 或 Git 索引。

## 基线与结论

- 审计基线为 `HEAD=3e1edd4` 加当前工作树未提交改动。审计期间工作树被其他进程并发修改；最终结论按 2026-10-22 02:05 后稳定的工作树执行。
- 关键快照 SHA-256：`pdf_workflow.py=053ad330...`、`_tmdextract.py=7027b3a...`、`_gradient_flow_quasi.py=d69e53f3...`、`_matching.py=bfc9703e...`、`_hybrid.py=0f0d89a...`、`_zr.py=bea3d3af...`。
- **当前实现不是可用于物理结论的最终胶子 TMD-PDF 链。** 现行 `run_tmd_pdf_chain` 已把状态写成 `diagnostic_only` / `physical_acceptance=False`，这一点是正确的；但原语、工作流和归档文档仍存在多处会把 diagnostic 误当成 physical 的风险。
- 当前 `pyqcd/testing/tmd/artifacts/work` cache 不能生成 PDF：唯一的 `P200` 输入是 `Nsample=1`，`c0_plateau_P200.npy` 全为 `NaN`，状态为 `statistically_unidentifiable`，且不存在准 PDF/匹配 PDF 产物。
- 历史 `final_run` / `extended_run` 只有 PNG、拟合报告和文档，没有可供本审计重放的 `.npy/.npz/.h5` 数值链；历史文档中的“PDF 达可交付标准”不能作为当前真实数据验证。

## 状态矩阵

| 环节 | 实现 | 测试证据 | 方案闭合 | 真实数据验证 |
|---|---|---|---|---|
| `exp(-ixPz z)` Fourier、Hermitian 延拓、因子 `1/(2πPz)` | 已实现 | 独立全轴 oracle 通过 | 在声明约定内闭合 | 当前 cache 无有效输入 |
| \(z=0\) 局域 spin-2 转换 | 已实现 | 解析契约通过；默认 `z0=NaN` fail-closed | 仍是有限流/有限格距诊断 | 未验证 |
| 梯度流端点因子 `c_parallel/c_perp/δm` | 已实现 | 原语与参考公式测试通过 | 仅一圈端点转换 | 工作流接入后无端到端测试 |
| hybrid 短/长距与 `Z_R` | 接口已实现 | hybrid 公式契约通过 | 否：未接 `Pz=0`、未拟合 `Z_R` | 否 |
| CS 两动量核 | 已实现 | 代数斜率/等号核测试通过 | 否：LO 比值 proxy，非 NLO 普适核 | 当前无两个有效 `Pz` |
| NLO gg 匹配 | 已实现 | `αs→0` 恒等、负 `ξ` 分支测试通过 | 否：无 soft、无 g/q mixing、支撑不足 | 否 |
| SFTX 系数 | 已实现 | 仅流时单位测试通过 | 否：局域 building block，不是本 TMD 匹配 | 否 |
| 连续/无穷动量外推 | 框架已实现 | 仅合成数据测试通过 | 否：工作流单格距/单主 `Pz`/单 `τ` | 否 |

## P0：阻止声明物理结果的问题

### P0-1 soft/rapidity 与胶子-夸克混合仍未测，匹配仍是 gg-only proxy

- 证据：工作流固定 `soft_factor=1.0`，并把缺失项明确写成 `soft_rapidity_not_measured`、`gg_only_no_gluon_quark_mixing`，见 `pyqcd/testing/tmd/pdf_workflow.py:482-505`。
- 证据：匹配元数据自报 `one_loop_gg_only_diagnostic`，见 `pyqcd/renorm/_tmdextract.py:449-458`。
- 物理影响：没有同几何 soft/rapidity subtraction，不能把结果称为 TMD-PDF；在 2+1 味 QCD 中省略 quark-singlet mixing 会改变胶子分布的尺度和混合演化。`soft=1` 不是“零软贡献”的测量，只是缺失。
- 参考边界：参考实现也明确写有 `singlet mixing omitted`、`single a`、`no controlled SFTE/continuum`，见 `refer/donghx/_Gradient_Flow_GluonPDF/lambda_fourier_quasi_matching_v2.py:300-309`。
- 最小 fail-closed：物理模式必须要求同流时间、同 staple 几何、同表示的 \(S_I(b)\) 以及 \(Z_{ij}\) 的 gg/gq 两分量；任一缺失只能写 `diagnostic`，不得生成 physical 名称。
- 应新增测试：`soft_factor=1` 在 physical 模式必须报错；已知 2×2 \(Z_{ij}\) 在 `αs→0` 必须精确等于单位矩阵；缺少 quark TMD 输入时禁止 gg-only 输出升级。
- 复现：`R1`。

### P0-2 固定 `τ=3`、单格距、单主 `Pz`，且 flow window 只记录不门控

- 证据：工作流固定 `conf='L24x72'`，主 PDF 只取 `next(iter(hR_store))`，`τ` 只作为输入参数；没有多 `τ`/多 `a`/多主动量外推，见 `pyqcd/testing/tmd/pdf_workflow.py:346-349,385-394`。
- 证据：`flow_scale_mask` 算出 `sqrt(8τ) < z`，但工作流仅把失败写成 reason，仍继续 Fourier 和匹配，见 `pyqcd/testing/tmd/pdf_workflow.py:491-505`。
- 证据：`τ=3` 时流半径 `sqrt(24)=4.899a`；`z/a=0..12` 的 mask 只有 `z/a>=5` 的非零点通过，`z/a=1..4` 全部失败。参考窗口要求见 `docs/胶子TMD梯度流连续极限.tex:1013`，连续极限要求见同文 `:760-772`。
- 物理影响：当前大部分小 `z` 非局域点处于流半径覆盖区；单 `τ=3a²` 不能把端点小流时转换当作受控 SFTX，也不能代替连续极限和无穷动量极限。
- 最小 fail-closed：物理模式只对通过 flow mask 的 `z` 做积分/匹配；`τ`、`a`、主 `Pz` 至少各两个并保存外推协方差与系统误差，否则只输出 diagnostic。
- 应新增测试：`sqrt(8τ)<z` 边界逐点断言；未通过 mask 的输入不得进入 Fourier；单 `a/Pz/τ` 不能设置 `physical_acceptance=True`。
- 复现：`R2`。

### P0-3 当前工作 cache 没有可识别的矩阵元，不能生成“最终 PDF”

- 证据：`c0_plateau_P200.npy` 全为 `NaN`，状态为 `statistically_unidentifiable`；`ratio_proton_P200.npy` 非有限，且没有 `quasi_tmd_pdf_*.npy`、`matched_tmd_pdf_*.npy` 或 `chain_status.json`。
- 证据：默认 `verify_pdf_workflow` 当前退出码为 1；`--allow-partial` 才给出 PARTIAL，明确表示不能声明物理 PDF。
- 物理影响：没有逐组态可识别矩阵元，就不存在可重放的 PDF 数值链；历史图片不能替代当前统计输入。
- 最小 fail-closed：保留当前 `diagnostic_only` 状态；在下游发布逻辑中要求 `chain_status.physical_acceptance=True` 才允许 `final`/`physical` 名称。
- 应新增测试：全 NaN/rank 0 plateau 必须阻止 PDF 产物生成；历史 PNG-only 目录不能通过真实数据验证门。
- 复现：`R3`。

## P1：会直接改变数值或方案含义的问题

### P1-1 匹配积分把 quasidistribution 的 `y` 支撑错误截到 `|y|<0.98`

- 证据：工作流使用 `x_grid_full = [-0.98,0.98]`，并把同一个网格同时传给 `x_grid` 和 `y_grid`，见 `pyqcd/testing/tmd/pdf_workflow.py:435,503-505`。
- 证据：匹配矩阵对超出 `y` 支撑的 `x/ξ` 直接丢弃，见 `pyqcd/renorm/_matching.py:151-164`；`tmd_matching_hybrid` 甚至强制 `len(y)==len(x)`，见 `pyqcd/renorm/_tmdextract.py:378-391`。
- 参考：参考实现用 `x∈[-1,1]`、`y∈[-3,3]`，以保留 quasi 支撑和负 `ξ` 支路，见 `refer/donghx/_Gradient_Flow_GluonPDF/lambda_fourier_quasi_matching_v2.py:232-233` 与 `match_unpolarized_quasi_from_extrap_v2.py:56-75`。
- 合成反例：在相同输入函数下，`y∈[-0.98,0.98]` 与 `y∈[-3,3]` 的结果在 `0.05<=x<=0.9` 最大绝对差为 `0.628`，正 `x` 曲线峰谷差从约 `4.28` 变为 `41.77`。这是直接的数值错误，不是“形状差异”。
- 最小修复：将输出 `x` 支撑和 convolution `y` 支撑分离；至少使用参考文献的宽 `y` 网格，并对 `|x|` 近零区域设显式 `xgap` NaN/不可计算 mask。
- 应新增测试：同解析输入下 `x/y` 独立支撑收敛；`y` 支撑加倍时结果稳定；负 `ξ` 与 `|x|<0.02` 的 mask 可追溯。
- 复现：`R4`。

### P1-2 CS 两动量核是 LO 单点比值 proxy，且拒绝合法符号变化

- 证据：`cs_kernel_two_momentum` 只取一个 `z_ref` 行，并要求比值严格为正，见 `pyqcd/renorm/_tmdextract.py:278-293`；工作流固定 `z_ref=1`，见 `pyqcd/testing/tmd/pdf_workflow.py:463-471`。
- 证据：参考公式使用 \(\ln|\phi(0,b;P_1)/\phi(0,b;P_2)|\) 并指出 NLO 需硬核修正，见 `docs/格点QCD中的TMD_PDF.tex:417-430`。当前实现对负比值直接抛 `ValueError`，没有取绝对值或复相位处理。
- 物理影响：CS 核可能在噪声统计下改变符号；拒绝负比值不是 fail-closed 的物理测量，而是丢失有效点。单 `z_ref`、两动量、无硬核修正也不足以称为普适 CS 核。`z_ref=1` 本身通常不满足 `sqrt(8τ)<z`。
- 最小修复：至少实现 `ln|ratio|` 的复相位明确定义、在多个满足 flow window 的 `z_ref` 上扫描并报告稳定性；NLO 使用硬核比值修正；只有稳定结果才升级为 CS measurement。
- 应新增测试：已知负比值应给出文档定义的有限核；`K` 等号核精确恢复；两/三 `Pz` 回归斜率与协方差一致；无效 ratio 与算法不可识别分开报告。
- 复现：`R5`。

### P1-3 hybrid/Z_R 代码存在但未进入工作流，当前实际是局域 spin-2 shape/幅度诊断

- 证据：`tmd_renormalize_hybrid` 和 `fit_ZR` 只在 `pyqcd/pipeline/_tmd9.py:932` 等位置定义，工作流没有调用；`rg` 只能找到定义而无生产调用。
- 证据：`pdf_workflow` 在 `pyqcd/testing/tmd/pdf_workflow.py:357-429` 走 `match_one` 端点转换，然后选择 `local_spin2` 或 `self_renormalize`，没有 `Pz=0` 的 hybrid 分母、没有 `Z_R` 拟合、没有短长距拼接。
- 物理影响：没有 Pz=0 分母和 Z_R，就没有文档 Eq. hybrid 所要求的长距自重整化；当前输出不能继承 `_hybrid.py` 的拼接语义。
- 最小 fail-closed：读取并校验同 b、同 tau、同几何的 `Pz=0` plateau，拟合 `Z_R` 并要求优化成功；否则 `hybrid` 模式必须拒绝，只允许 `local_spin2_diagnostic`。
- 应新增测试：缺少 Pz=0 时拒绝 hybrid；拼接点逐 sample/b 连续；非平坦 Z_R 必须改变长距结果；Z_R 拟合失败不得被当作可用。
- 复现：`R6`。

### P1-4 端点转换与匹配使用了不一致的 `a`/`αs`，绝对幅度误差未配对传播

- 证据：`convert_c0` 硬编码 `a_fm=0.1053`、`αs=0.25`，见 `pyqcd/testing/tmd/pdf_workflow.py:357-371`；网格和 `pz_to_gev` 使用 `a_len_set['L24x72']=0.105`，见 `pyqcd/renorm/_ensembles.py:8-17`。匹配层又从 `A_s_run(2.0)` 取一圈耦合，当前值为 `0.322787`，而文档引用常数为 `0.296`。
- 证据：`normalization='local_spin2'` 使用一个由 C2 窗口中位数得到的标量 `p_t` 乘到所有 sample，见 `pyqcd/testing/tmd/pdf_workflow.py:390-425`；没有为每个 jackknife sample 保持能量与矩阵元的联合索引。
- 物理影响：`c_perp`、线质量项和绝对 `xg` 幅度会因约 29% 的耦合差异以及 `a` 的不一致而系统偏移；统计误差也不包含能量-矩阵元相关性。
- 最小修复：引入单一 `LatticeContract(a, Pz_gev, alpha_s(mu), nf, mu)`，由同一对象驱动转换、Fourier 和匹配；能量必须在同一重采样索引上传播。物理模式禁止未记录来源的 fallback。
- 应新增测试：修改 `a`/`αs` 同时只改变声明的端点/匹配因子；共享 jackknife 索引的合成数据恢复预期联合误差；`p_t` fallback 不能通过 physical gate。
- 复现：`R7`。

### P1-5 当前输入为实数时只产生偶 `x` 分支，signed-x 只存在于网格而非数据

- 证据：`quasi_tmd_pdf` 对实数 `h_R` 只做余弦积分，见 `pyqcd/renorm/_tmdextract.py:49-50,114-130`；当前工作流从 `c0` 经 `self_renormalize` 得到的输入是实数。
- 证据：`x_grid_full` 的负半轴只让匹配矩阵接受负 `ξ`，但不会从实偶输入制造反胶子奇支路；`_hybrid.hR_x` 也只有 `cos`，见 `pyqcd/renorm/_hybrid.py:198-230`。对复数输入，`hR_x(+x)` 与 `hR_x(-x)` 完全相等。
- 物理影响：如果目标是 signed-x/反夸克支路的完整胶子 TMD，当前 `x<0` 不能作为独立物理信息；只能作为已声明偶通道的数值延拓。
- 最小修复：采集并保存 `+z/-z` 同几何矩阵元，保留复数虚部，严格使用 `exp(-ixPz z)`；若只做偶通道，输出名和 metadata 必须明确 `even-only`。
- 应新增测试：实偶输入与复奇输入分别命中不同对称性；`h(-z)=h(z)*` oracle；禁止用负 `x` 网格替代反胶子数据。
- 复现：`R8`。

### P1-6 `fit_ZR` 不检查优化器成功状态，失败结果会被下游当作参数

- 证据：Scipy 分支直接 `return res.x`，没有检查 `res.success`、`fun`、参数边界或 Hessian；iminuit 分支也没有 `m.valid` 检查，见 `pyqcd/renorm/_zr.py:534-560`。
- 合成反例：把 `scipy.optimize.minimize` mock 成 `success=False`、非有限 `fun`，`fit_ZR` 仍返回参数 `[1,2,3,...]`。
- 物理影响：失败拟合会产生“数值有限但物理无效”的 Z_R，进而污染长距 hybrid 结果；`fit_ZR_samples` 只有抛异常才标记 NaN，无法识别静默失败。
- 最小修复：定义结构化 fit status，必须检查成功、有限、边界、协方差/秩；sample loop 只汇总成功样本，并把失败比例作为硬门。
- 应新增测试：mock `success=False` 必须拒绝；边界/奇异 Hessian 必须标记不可识别；单坏样本和全坏样本状态不同。
- 复现：`R9`。

## P2：边界、状态和测试治理问题

### P2-1 diagnostic 命名与历史“可交付”叙述仍不一致

- 证据：模块顶部仍写“全物理链”和“NLO 匹配 \(\to\) 光锥 TMD-PDF”，见 `pyqcd/testing/tmd/pdf_workflow.py:1-17`；实际函数已写 `diagnostic_only`，见 `:323-329`。
- 证据：历史归档 `docs/from-logs/logs/dev5_2/analy_test9_20260822.tex:684` 写“PDF 达可交付标准”，与当前 `soft=1`、gg-only、单 a、无连续极限的状态冲突。
- 影响：自动化用户会按文件名/标题读取 `matched_tmd_pdf_xg.npy`，把 diagnostic 当 final。
- 最小修复：把输出改为 `diagnostic_*`，或将 final 发布限制在 `physical_acceptance=True`；归档文档加“历史/未验证”醒目标记。
- 应新增测试：`physical_acceptance=False` 时禁止 `final` 命名；报告生成器拒绝把 diagnostic 写成“物理 PDF”。

### P2-2 `_hybrid` 的符号/边界处理仍不完整

- 证据：`hR_z_Pz` 使用 `mask = z_ < zs`，而文档公式要求 `|z|<z_s`，见 `pyqcd/renorm/_hybrid.py:52-58` 与 `docs/胶子TMD梯度流连续极限.tex:607-611`。
- 证据：只有短距或只有长距时 `eta_s` 索引会抛 `IndexError`；复数 `hR` 虽保留 dtype，但 `hR_x` 仍然只用余弦；`fit_hR_lambda` 又显式拒绝复数，见 `pyqcd/renorm/_hybrid.py:55-62,102-106,198-230`。
- 物理影响：带符号 `z`、helicity/复通道和边界 `z_s` 会产生错误/不可诊断行为；当前非负 `z` 数据暂时绕过了第一项。
- 最小修复：验证 `z>=0` 或统一用 `abs(z)`；对空 mask 抛明确 `ValueError`；将 even-only Fourier 写成显式 API，另设 Hermitian/odd API。
- 应新增测试：负 `z` 等价于正 `z` 的拼接；全短/全长 mask 明确拒绝；复数 Hermitian oracle；`z_s` 两端边界。

### P2-3 文档公式与实现存在会误导重写的差异

- 证据：主文档的 Z_R 线性项写成 `k z/a ln(aΛ)`，见 `docs/胶子TMD梯度流连续极限.tex:618`；实现和参考源码实际使用 `k z/(a ln(aΛ))`，见 `pyqcd/renorm/_zr.py:179` 与 `refer/huangcl/_0xxx/fit_zr_new.py:104,144`。
- 证据：部分当前报告写 CS 默认 `|K|<=3`，但 `cs_kernel_two_momentum` 默认 `k_clip=None`，见 `pyqcd/renorm/_tmdextract.py:245-293`。
- 影响：按文档重写会产生数量级不同的 Z_R，或错误地以为 CS 已自动截断。
- 最小修复：修正 LaTeX 公式和默认值叙述，增加由公式直接计算的数值 oracle，而不是只断言“有限”。
- 应新增测试：`th_ZR` 对参考点逐项数值比对；文档关键默认值与函数签名/行为的契约检查。

### P2-4 当前验证入口与技能说明不完全一致

- 证据：`skills/pyqcd-tmd-algorithm/references/validation.md` 要求 `python -B -m pyqcd.testing._tmd9_hybrid_contract`；当前该模块不存在，实际可运行的是 `pyqcd.testing._tmd_hybrid_contract`。
- 稳定性：`_tmd_hybrid_contract` 在最终工作树运行 `44/44 OK`；该测试主要覆盖缓存契约/hybrid 接线，不证明物理匹配。
- 最小修复：恢复或重命名正式验证入口，避免技能文档运行一个不存在的模块；在测试报告中分开列出“契约通过”和“物理门通过”。

## 合成极限核对

- **`αs→0`**：`match_one` 在 `τ>0` 的非局域点保持输入；`tmd_matching_hybrid` 在 patch `A_s_run=0` 时 `max|out-input|=0`。这证明单位核结构，不证明非零阶匹配正确。
- **等号 CS 核**：构造 `K=0,1,5` 的 `c0` 比值，`cs_kernel_two_momentum` 精确恢复对应斜率；证明代数斜率，不证明物理 CS 提取。
- **对称性/Fourier**：正半轴 Hermitian 数据对全轴独立积分通过；实输入给出 `q(+x)=q(-x)`，复输入给出不同 `+x/-x`，因此 signed-x 必须依赖复数据。
- **零流**：`coefficients(τ=0/-1)` 正确拒绝；零流端点公式不可定义，不能把 `τ=3` 的诊断当连续流极限。
- **`z=0`**：默认 `z0_policy='nan'` fail-closed；`spin2_local` 的局域因子有解析测试。工作流虽然已接入该 policy，但没有端到端结果可供验证。
- **两端 `z_s`/hybrid**：公式拼接测试通过，但仅限合成、正 `z`、实通道。

## 复现命令

以下命令均为只读；测试使用 `-B` 避免写字节码。

`R1`：检查 soft/mixing 缺口与 gg-only 状态。

```bash
rg -n "soft_factor=1\\.0|gg_only_no_gluon_quark_mixing|cs_kernel_not_measured" \
  pyqcd/testing/tmd/pdf_workflow.py pyqcd/renorm/_tmdextract.py
```

`R2`：检查固定流时、flow window 与单动量路径。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.renorm import flow_scale_mask
print('radius/a =', np.sqrt(24.0))
print('z/a      =', np.arange(13).tolist())
print('mask     =', flow_scale_mask(np.arange(13.0), 3.0).astype(int).tolist())
PY
rg -n "main_tag|x_grid_full|flow_mask|fit_hR_PDF_extrap|single_lattice" \
  pyqcd/testing/tmd/pdf_workflow.py
```

`R3`：验证当前 cache 不可识别且没有 PDF 产物。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import os, numpy as np
root='pyqcd/testing/tmd/artifacts/work/analysis/tmd_ratio'
a=np.load(root+'/c0_plateau_P200.npy')
with np.load(root+'/c0_plateau_status_P200.npz', allow_pickle=False) as d:
    print('shape=',a.shape,'all_nan=',bool(np.isnan(a).all()))
    print('status=',str(d['plateau_status']))
print('pdf_files=',[f for f in os.listdir(root) if 'pdf_' in f or f.startswith('matched_')])
PY
PYTHONDONTWRITEBYTECODE=1 python -B -m pyqcd.testing.tmd.verify_pdf_workflow \
  --skip-flow pyqcd/testing/tmd/artifacts/work
```

`R4`：比较窄/宽 convolution 支撑。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.renorm import tmd_matching_hybrid
x=np.linspace(-.98,.98,256)
yn=x.copy(); yw=np.linspace(-3,3,256)
q=lambda y: np.exp(-y*y)*(1+.25*np.tanh(1.5*y))
args=dict(pz_gev=2.,pz_scale=2.,cs_kernel=.2,soft_factor=1.,lambda_s_fm=.3)
_,a=tmd_matching_hybrid(x,y_grid=yn,x_tmd=q(yn),**args)
_,b=tmd_matching_hybrid(x,y_grid=yw,x_tmd=q(yw),**args)
r=(x>=.05)&(x<=.9)
print('max|narrow-wide|=',float(np.max(np.abs(a[r]-b[r]))))
print('ptp narrow/wide=',float(np.ptp(a[x>0])),float(np.ptp(b[x>0])))
PY
```

`R5`：检查文档 `ln|ratio|` 与当前正值限制。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.renorm import cs_kernel_from_ratio
h1=np.array([-2.,3.]); h2=np.array([1.,2.])
print('expected log|r|/log4/2 =',float(np.log(abs(h1[0]/h2[0]))/np.log(2.)))
try: cs_kernel_from_ratio(h1,h2,4.,2.)
except Exception as e: print(type(e).__name__, str(e))
PY
```

`R6`：确认 hybrid/Z_R 没有生产调用。

```bash
rg -n "tmd_renormalize_hybrid\\(|fit_ZR\\(|sftx_gluon_matching_coeff" \
  pyqcd/testing/tmd/pdf_workflow.py pyqcd/pipeline/_tmd9.py
```

`R7`：检查耦合、格距和匹配标度不一致。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
from pyqcd.renorm import alpha_s
print('A_s_run default alpha_s(2) =',alpha_s(2.0))
print('document quoted            = 0.296')
PY
rg -n "0\\.1053|0\\.25|A_s_run|pz_to_gev|p_t_gev" \
  pyqcd/testing/tmd/pdf_workflow.py pyqcd/renorm/_matching.py pyqcd/renorm/_zr.py
```

`R8`：检查实/复 signed-x 行为。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import numpy as np
from pyqcd.renorm import quasi_tmd_pdf, hR_x
z=np.linspace(0,.8,17); h=np.exp(-z)[:,None]
_,q=quasi_tmd_pdf(h,z,[.2],2.,x_grid=np.array([.37,-.37]),n_pts=4097)
print('real q(+/-), diff =',q[0],q[1],q[0]-q[1])
z2=np.linspace(0,.8,9); hr=1+.3j*z2
o=hR_x(np.array([.37,-.37]),4,100.,1.,1.,10.,zs=.4,
       z_data=z2,hR_pz=hr,hR_0=np.ones_like(z2),
       zr_fit=np.ones_like(z2),conf='L24x72')
print('hR_x(+/-), diff =',o[0],o[1],o[0]-o[1])
PY
```

`R9`：检查 Z_R 失败优化器仍返回值。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pyqcd.renorm import fit_ZR
z=np.array([.15,.3,.45,.6,.75])
ds={'z':z,'loghB':np.zeros_like(z),'c_inv':np.eye(len(z)),'a':.53}
bad=SimpleNamespace(x=np.arange(21.)+1,success=False,fun=np.nan,message='bad')
with patch('scipy.optimize.minimize',return_value=bad):
    print(fit_ZR(np.zeros(21),[ds],2.,use_iminuit=False)[:3])
PY
```

## 已执行验证

- `python -B -m pyqcd.testing._quasi_tmd_fourier_contract`：13/13 PASS。覆盖 Hermitian 全轴、Fourier 因子、`x` 依赖 `ζ_z`、负 `ξ` 网格和 `αs→0` identity。
- `python -B -m pyqcd.testing._gradient_flow_quasi_policy_contract`：7/7 PASS。覆盖 z=0 默认 NaN、局域 spin-2、helicity minus/legacy plus 和 window policy。
- `python -B -m pyqcd.testing._renorm_numerical_guards_contract`：26/26 PASS。覆盖协方差、复数、kernel 输入和失败边界。
- `python -B -m pyqcd.testing._tmd_hybrid_contract`：44/44 PASS。覆盖 hybrid 拼接/缓存契约；不构成物理闭合。
- `python -B -m pyqcd.testing._tmd_soft_factor_ze_contract`：8/8 PASS。仅验证矩形圈接口和 `sqrt(S)` 消费，不验证 TMD soft 数据。
- `python -B -m pyqcd.testing._matching_grid_contract`：4/4 PASS。仅验证网格/单位核/负 `ξ` 局部公式。
- `python -B -m pyqcd.testing._gradient_flow_gluon_contract`：12/12 PASS。覆盖流/算符/端点 primitive，不覆盖当前 PDF 工作流的联合数值正确性。
- `python -B -m pyqcd.testing._sftx_units_contract`：2/2 PASS。只证明 `tau/a_fm` 到 `GeV^-2` 的量纲换算，没有独立物理系数 oracle。
- 当前 `verify_pdf_workflow --skip-flow`：退出 1；`--allow-partial --skip-flow`：PARTIAL，明确不能声明物理 PDF。
- 当前工作树没有稳定的正式 `_tmd9_hybrid_contract` 入口；可用的 `_tmd_hybrid_contract` 已单独运行。

## 最终判断

可以称为 **已实现并且有受控测试** 的是：Fourier 符号/因子、Hermitian 正半轴延拓、z=0 局域转换的 primitive、hybrid 公式的逐点代数、匹配核的 `αs→0` 与负 `ξ` 分支、部分 cache 契约。

不能称为 **方案闭合** 的是：soft/rapidity、CS 普适核、g/q mixing、hybrid `Z_R` 的真实数据来源、flow/SFTX 连续窗口和多尺度外推。

不能称为 **真实数据验证** 的是：当前 single-config `Nsample=1` cache、历史只有图片的 `final_run`、以及所有没有同一组态索引和误差账本的 PDF 图。

在完成 P0 和 P1-1/P1-2/P1-3 之前，任何 `xg`、CS 核或“matched TMD-PDF”曲线都应保留 `diagnostic` 标签，不能用于胶子 TMD-PDF 的物理结论。
