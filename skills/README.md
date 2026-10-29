# PyQCD 领域 Skills

本目录提供 PyQCD 格点 QCD 研究的直接领域技能。Agent 先按“功能”筛选最小技能集，再
读取对应 `SKILL.md`；较长的 API、几何和验收材料按需读取各技能的 `references/`。

## 直接技能目录

| Skill 目录 | 创建者 | 功能 | 最近一次更改时间 | 入口 |
|---|---|---|---|---|
| [`pyqcd-analysis/`](pyqcd-analysis/) | `root` | 运行已有数据的物理量提取和图表；统计纪律与谱模型分别由专门技能定义。 | `2026-10-13` | [`SKILL.md`](pyqcd-analysis/SKILL.md) |
| [`pyqcd-bare-matrix/`](pyqcd-bare-matrix/) | `root` | 约束 physical Tmin、Ratio/Sumratio 拟合、AIC 权重、完整 bootstrap 覆盖与 fallback 状态。 | `2026-10-13` | [`SKILL.md`](pyqcd-bare-matrix/SKILL.md) |
| [`pyqcd-blending/`](pyqcd-blending/) | `root` | 定义低模加随机正交补的 blended all-to-all 估计量、elementals、perambulator 与两点收缩。 | `2026-10-13` | [`SKILL.md`](pyqcd-blending/SKILL.md) |
| [`pyqcd-conventions/`](pyqcd-conventions/) | `root` | 统一 γ 矩阵、轴序、单位、边界、Fourier/basis 符号、复数和结果状态契约。 | `2026-10-13` | [`SKILL.md`](pyqcd-conventions/SKILL.md) |
| [`pyqcd-distillation/`](pyqcd-distillation/) | `root` | 审计 LapH/Wick/perambulator/GEVP/MPI/I/O 因果链及当前实现证据边界。 | `2026-10-13` | [`SKILL.md`](pyqcd-distillation/SKILL.md) |
| [`pyqcd-docs/`](pyqcd-docs/) | `root` | 编写和验收中文 LaTeX、analy/pure 报告及 PDF 版式与源码证据。 | `2026-08-31 10:15:20 +08:00` | [`SKILL.md`](pyqcd-docs/SKILL.md) |
| [`pyqcd-enhanced-interpolator/`](pyqcd-enhanced-interpolator/) | `root` | 构造和审查 boosted-hadron 增强内插算符、Euclidean γ± 与 direct-minus-exchange 缩并。 | `2026-10-13` | [`SKILL.md`](pyqcd-enhanced-interpolator/SKILL.md) |
| [`pyqcd-gauge/`](pyqcd-gauge/) | `root` | 计算 Wilson/Polyakov 圈、静态势、拓扑荷、Wilson flow 和链接涂抹。 | `2026-10-13` | [`SKILL.md`](pyqcd-gauge/SKILL.md) |
| [`pyqcd-gluon-renorm/`](pyqcd-gluon-renorm/) | `root` | 区分 bare、gauge-fixed 与 renormalized 胶子可观测量，并约束 Clover、Fourier、EMT/current 契约。 | `2026-10-13` | [`SKILL.md`](pyqcd-gluon-renorm/SKILL.md) |
| [`pyqcd-helicity-renorm/`](pyqcd-helicity-renorm/) | `root` | 显式审查 legacy helicity/EMT matching、R12/R21 放置与 coupled running 来源冲突。 | `2026-10-13` | [`SKILL.md`](pyqcd-helicity-renorm/SKILL.md) |
| [`pyqcd-hisq-recover/`](pyqcd-hisq-recover/) | `root` | 将 staggered/HISQ 单分量传播子重构为 naive 四旋量 Wilson-like 存储，并约束 producer metadata。 | `2026-10-13` | [`SKILL.md`](pyqcd-hisq-recover/SKILL.md) |
| [`pyqcd-infra/`](pyqcd-infra/) | `root` | 处理后端、GPU 计时、HDF5/ASCII/VdV/VVV I/O、MPI/显存规划，以及对照数据体积盘点、分层取样和断点下载。 | `2026-10-25` | [`SKILL.md`](pyqcd-infra/SKILL.md) |
| [`pyqcd-momentum-smear/`](pyqcd-momentum-smear/) | `root` | 定义 link-phased Wuppertal 动量涂抹、source/sink/sequential one-smear 与相位链。 | `2026-10-13` | [`SKILL.md`](pyqcd-momentum-smear/SKILL.md) |
| [`pyqcd-physics-correlator/`](pyqcd-physics-correlator/) | `root` | 将观测量推导为算符、关联函数、Wick、蒸馏 basis/传播子交接和 einsum。 | `2026-10-13` | [`SKILL.md`](pyqcd-physics-correlator/SKILL.md) |
| [`pyqcd-physics-spectrum/`](pyqcd-physics-spectrum/) | `root` | 权威定义谱分解、色散似然、backward 态和两点/三点拟合模板。 | `2026-08-31 09:11:23 +08:00` | [`SKILL.md`](pyqcd-physics-spectrum/SKILL.md) |
| [`pyqcd-pipeline/`](pyqcd-pipeline/) | `root` | 编排蒸馏管线、严格 OPE artifact contract、GPU 完成计时、原子持久化、断点续跑、守卫和环境快照。 | `2026-08-31 10:46:26 +08:00` | [`SKILL.md`](pyqcd-pipeline/SKILL.md) |
| [`pyqcd-propagator/`](pyqcd-propagator/) | `root` | 用 PyQUDA 求解 Wilson/Clover 传播子、顺序源和协变非局部位移。 | `2026-10-13` | [`SKILL.md`](pyqcd-propagator/SKILL.md) |
| [`pyqcd-quark-pdf/`](pyqcd-quark-pdf/) | `root` | 导航连通夸克梯度流准 PDF 的 4D flow、ringed normalization、SFTE、matching 与 u-d 边界。 | `2026-10-13` | [`SKILL.md`](pyqcd-quark-pdf/SKILL.md) |
| [`pyqcd-quark-renorm/`](pyqcd-quark-renorm/) | `root` | 分级管理夸克双线性 RI-to-MSbar conversion/running 与 profile provenance。 | `2026-10-13` | [`SKILL.md`](pyqcd-quark-renorm/SKILL.md) |
| [`pyqcd-ri-renorm/`](pyqcd-ri-renorm/) | `root` | 验证 RI'/MOM 或 RI/SMOM 动量学、独立双腿截肢与 raw quark-bilinear Z factors。 | `2026-10-13` | [`SKILL.md`](pyqcd-ri-renorm/SKILL.md) |
| [`pyqcd-statistics/`](pyqcd-statistics/) | `root` | 统一 jackknife/bootstrap、协方差、模型 Jacobian 可辨识性、拟合窗口和统计诊断纪律。 | `2026-10-13` | [`SKILL.md`](pyqcd-statistics/SKILL.md) |
| [`pyqcd-tmd-algorithm/`](pyqcd-tmd-algorithm/) | `root` | 约束梯度流核子胶子 TMD 的几何、重整化、匹配、外推、真实 GPU 生产、缓存和验证门。 | `2026-10-29` | [`SKILL.md`](pyqcd-tmd-algorithm/SKILL.md) |
| [`pyqcd-tmd-chain/`](pyqcd-tmd-chain/) | `root` | 导航梯度流、OPE、断连、Z_R/混合、Fourier/CS/匹配到连续极限全链。 | `2026-10-13` | [`SKILL.md`](pyqcd-tmd-chain/SKILL.md) |

## 运维技能

| Skill 目录 | 创建者 | 功能 | 最近一次更改时间 | 入口 |
|---|---|---|---|---|
| [`form/`](form/) | `root` | 审计并维护 PyQCD 命名、目录、交付格式、具名例外和本地验证入口。 | `2026-09-28 04:24:32 +08:00` | [`SKILL.md`](form/SKILL.md) |

## 维护边界

- 表中 23 个 `pyqcd-*` 目录及其 reference 由当前用户 `root` 维护；新增或更新入口时同步
  更新本表。
- `form/` 由当前用户 `root` 维护，是仓库运维技能，不属于上述 23 个物理领域技能。
- `LQCD_Skills/`、`donghx/` 与 `sush/` 是只读外来参考树；其内容只由对应
  `pyqcd-*` 自包含技能吸收，不在本表转移所有权，也不原地改写。
- [`sush/README.md`](sush/README.md) 单独管理 `sush/lqcddb`；其创建者为 `sush`，本目录
  只读借鉴，不在本表转移或冒领所有权。
- 共享规范与调用关系见 [`AGENTS.md`](AGENTS.md)；本 README 是技能发现目录。
