# AGENTS.md — PyQCD 项目技能库

本文件管理 `/root/PyQCD/skills` 下由当前用户维护的 PyQCD 领域技能。直接技能目录和
入口链接以 [`README.md`](README.md) 为发现目录；本文件维护职责边界、调用关系和共同
约定。上层项目运行命令与物理研究背景见 `/root/PyQCD/AGENTS.md`。

## 技能注册表

| 技能 | 单一职责 | 典型触发 |
|---|---|---|
| `pyqcd-conventions` | γ、轴序、单位、边界、Fourier/basis 符号、元数据和证据状态 | 约定冲突、shape/单位/符号不明 |
| `pyqcd-physics-correlator` | 观测量 → 算符 → 关联函数 → Wick → 传播子/蒸馏交接 → einsum | 算符、2pt/3pt、Wick、VdV/VVV、矩阵元 |
| `pyqcd-physics-spectrum` | 权威定义关联函数 → 谱分解 → 能量/重叠/拟合模板 | backward 态、激发态、谱式 |
| `pyqcd-propagator` | PyQUDA 组态、Dirac 求解、源、顺序源和 covDev | 求传播子、求逆、顺序源 |
| `pyqcd-gauge` | 纯规范路径、Wilson/Polyakov 圈、拓扑、流和涂抹 | 只有规范链接的观测量 |
| `pyqcd-statistics` | 重采样、协方差、SVD、窗口和统计诊断 | jackknife/bootstrap、相关拟合 |
| `pyqcd-analysis` | 已有数据的分析运行、物理量提取和图表；不定义谱模型或统计纪律 | ratio、有效质量、E0、FH、c0 |
| `pyqcd-infra` | backend、设备/精度、I/O、MPI 元任务和显存规划 | torch、HDF5、ASCII、VdV/VVV、OOM |
| `pyqcd-pipeline` | 九步运行编排、守卫、断点续跑和一致性验证 | test0、test9、基线复现 |
| `pyqcd-tmd-chain` | 梯度流胶子 TMD-PDF 的端到端导航 | 全链规划、阶段关系、状态判断 |
| `pyqcd-tmd-algorithm` | TMD 几何、重整化、匹配、外推和物理验证门 | 算法实现、staple、soft、CS、匹配 |
| `pyqcd-docs` | 中文 LaTeX、源码证据、PDF 和报告验收 | analy、pure、xelatex、版式 |

## 调用链

```text
pyqcd-conventions ─────────────────────────────────────────────┐
       │                                                       │
       ├→ pyqcd-physics-correlator → pyqcd-propagator ────────┤
       │             └→ pyqcd-physics-spectrum                │
       │                         └→ pyqcd-statistics           │
       ├→ pyqcd-gauge → pyqcd-tmd-chain → pyqcd-tmd-algorithm ─┤
       │                                                       │
       └───────────────────────→ pyqcd-analysis ←─────────────┤
                                  ↑             ↑              │
                         pyqcd-statistics   pyqcd-pipeline ← pyqcd-infra
                                  │             │
                                  └─────────────┴→ pyqcd-docs
```

读图规则：`conventions` 是共享先验；`correlator` 决定物理对象，`propagator` 生产数据，
`spectrum` 定义谱模型，`statistics` 定义统计门，`analysis` 调用并呈现数据；`gauge` 提供纯规范基础，`tmd-chain` 导航全链，
`tmd-algorithm` 承担 TMD 的实现和验收细节；`infra` 只提供平台底座，`pipeline` 只编排，
`docs` 只成文交验。

## 统一入口规范

每个直接技能的 `SKILL.md` 必须包含：

1. YAML frontmatter：`name` 与目录一致，`description` 以 `Use when...` 开头，只写触发
   条件，名称使用小写字母/数字/连字符；
2. 目的与边界：说明它负责什么、明确不负责什么；
3. 最小工作流程：可执行、按依赖排序；
4. 常见错误或验收门；
5. 与相邻技能的交接；
6. 超过约 100 行的 API、公式或长例文下沉到本技能 `references/`，入口只保留索引。

共同内容只保留一个权威来源：轴序/单位/边界/证据状态写入 `pyqcd-conventions`，
重采样/协方差/拟合纪律写入 `pyqcd-statistics`，后端/I/O/MPI 写入 `pyqcd-infra`。
其他技能通过名称路由，不复制整段规则；TMD 总览与 TMD 实现契约分离。

## 执行与验证约定

- 多步骤任务先列 TODO，完成一个子步骤立即更新；本库无 pytest 依赖，结构检查优先使用
  官方 `/root/.codex/skills/.system/skill-creator/scripts/quick_validate.py`（逐个
  `pyqcd-*` 目录运行）。
- 证据以当前源码、命令输出、数值断言和产物为准；引用 API 或文件前先核实存在性。
- 修改技能时使用最小范围、保留既有物理边界，不 import `refer/` 或 `examples/` 作为运行依赖。
- 入口文档目标小于 500 行；格式变更后检查 fenced block、路径、名称、表格和 README 一致性。
- 长任务日志使用当前工作目录的 `.X.<YYYY-MM-DD-HH-MM-SS>.log`，确认不入库；报告编译
  产物按 `pyqcd-docs` 规范验收。

## 所有权与只读边界

本文件和根 [`README.md`](README.md) 仅登记直接 `pyqcd-*` 技能。`sush/AGENTS.md`、
`sush/README.md` 及 `sush/lqcddb` 属于 `sush`，本任务只读借鉴，不修改、不重命名、不
更新创建者或目录条目。外来参考代码同样保持只读；需要吸收时在 PyQCD 自有技能中
自包含改写并注明验证边界。

## 维护清单

新增、修改或拆分直接技能时必须同步：

- `README.md` 的一条目录记录（创建者、功能、更新时间、入口）；
- 本文件的注册表和调用链（只有职责或关系变化时更新对应行）；
- 新增 `references/` 的相对链接与文件存在性；
- 官方入口校验、Markdown 结构检查、`git diff --check` 和未跟踪文件清单。
