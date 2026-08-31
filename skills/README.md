# PyQCD 领域 Skills

本目录提供 PyQCD 格点 QCD 研究的直接领域技能。Agent 先按“功能”筛选最小技能集，再
读取对应 `SKILL.md`；较长的 API、几何和验收材料按需读取各技能的 `references/`。

## 直接技能目录

| Skill 目录 | 创建者 | 功能 | 最近一次更改时间 | 入口 |
|---|---|---|---|---|
| [`pyqcd-analysis/`](pyqcd-analysis/) | `root` | 运行已有数据的物理量提取和图表；统计纪律与谱模型分别由专门技能定义。 | `2026-08-31 10:42:08 +08:00` | [`SKILL.md`](pyqcd-analysis/SKILL.md) |
| [`pyqcd-conventions/`](pyqcd-conventions/) | `root` | 统一 γ 矩阵、轴序、单位、边界、Fourier/basis 符号、复数和结果状态契约。 | `2026-08-31 08:47:37 +08:00` | [`SKILL.md`](pyqcd-conventions/SKILL.md) |
| [`pyqcd-docs/`](pyqcd-docs/) | `root` | 编写和验收中文 LaTeX、analy/pure 报告及 PDF 版式与源码证据。 | `2026-08-31 10:15:20 +08:00` | [`SKILL.md`](pyqcd-docs/SKILL.md) |
| [`pyqcd-gauge/`](pyqcd-gauge/) | `root` | 计算 Wilson/Polyakov 圈、静态势、拓扑荷、Wilson flow 和链接涂抹。 | `2026-08-31 10:52:42 +08:00` | [`SKILL.md`](pyqcd-gauge/SKILL.md) |
| [`pyqcd-infra/`](pyqcd-infra/) | `root` | 处理 numpy/cupy/torch 后端、GPU 计时、HDF5/ASCII/VdV/VVV I/O、MPI 与显存规划。 | `2026-08-31 10:15:20 +08:00` | [`SKILL.md`](pyqcd-infra/SKILL.md) |
| [`pyqcd-physics-correlator/`](pyqcd-physics-correlator/) | `root` | 将观测量推导为算符、关联函数、Wick、蒸馏 basis/传播子交接和 einsum。 | `2026-08-31 10:58:25 +08:00` | [`SKILL.md`](pyqcd-physics-correlator/SKILL.md) |
| [`pyqcd-physics-spectrum/`](pyqcd-physics-spectrum/) | `root` | 权威定义谱分解、色散似然、backward 态和两点/三点拟合模板。 | `2026-08-31 09:11:23 +08:00` | [`SKILL.md`](pyqcd-physics-spectrum/SKILL.md) |
| [`pyqcd-pipeline/`](pyqcd-pipeline/) | `root` | 编排蒸馏管线、严格 OPE artifact contract、GPU 完成计时、原子持久化、断点续跑、守卫和环境快照。 | `2026-08-31 10:46:26 +08:00` | [`SKILL.md`](pyqcd-pipeline/SKILL.md) |
| [`pyqcd-propagator/`](pyqcd-propagator/) | `root` | 用 PyQUDA 求解 Wilson/Clover 传播子、顺序源和协变非局部位移。 | `2026-08-31 02:27:30 +08:00` | [`SKILL.md`](pyqcd-propagator/SKILL.md) |
| [`pyqcd-statistics/`](pyqcd-statistics/) | `root` | 统一 jackknife/bootstrap、协方差、模型 Jacobian 可辨识性、拟合窗口和统计诊断纪律。 | `2026-08-31 09:11:23 +08:00` | [`SKILL.md`](pyqcd-statistics/SKILL.md) |
| [`pyqcd-tmd-algorithm/`](pyqcd-tmd-algorithm/) | `root` | 约束梯度流核子胶子 TMD 的几何、重整化、匹配、外推、缓存和验证门。 | `2026-08-31 06:44:19 +08:00` | [`SKILL.md`](pyqcd-tmd-algorithm/SKILL.md) |
| [`pyqcd-tmd-chain/`](pyqcd-tmd-chain/) | `root` | 导航梯度流、OPE、断连、Z_R/混合、Fourier/CS/匹配到连续极限全链。 | `2026-08-31 02:27:30 +08:00` | [`SKILL.md`](pyqcd-tmd-chain/SKILL.md) |

## 维护边界

- 表中 12 个 `pyqcd-*` 目录及其 reference 由当前用户 `root` 维护；新增或更新入口时同步
  更新本表。
- [`sush/README.md`](sush/README.md) 单独管理 `sush/lqcddb`；其创建者为 `sush`，本目录
  只读借鉴，不在本表转移或冒领所有权。
- 共享规范与调用关系见 [`AGENTS.md`](AGENTS.md)；本 README 是直接技能的发现目录。
