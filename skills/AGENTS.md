# AGENTS.md — PyQCD 项目技能库（/root/PyQCD/skills）

PyQCD 专属领域技能库。每个技能一个子目录，含 `SKILL.md`（frontmatter + 正文）；
部分技能带 `reference/` 例文。服务于格点 QCD 研究仓库 /root/PyQCD
（核心目标：梯度流重整化方案下核子胶子 TMD-PDF）。
来源：上游 LQCD 五技能吸收改写（原 .opencode/skills，已并入后删除）
+ logs/ 实战套件约定沉淀（stab1、test0 系列、test6–test9、dev5–dev7）。

## 技能注册表

| 技能 | 用途 | 关键触发 |
|---|---|---|
| `pyqcd-physics-correlator` | 观测量→算符→关联函数→Wick 缩并→传播子清单→einsum 推理链 | 强子质量/矩阵元/形状因子/"需要哪些关联函数" |
| `pyqcd-physics-spectrum` | 谱分解：完备关系/重叠因子/backward 态→拟合函数模板 | 谱分解/激发态污染/"用什么拟合函数" |
| `pyqcd-propagator` | PyQUDA 传播子求解：组态加载/Clover/multigrid/源构造/顺序源/covDev | "解传播子"/"调 PyQUDA"/3pt sequential |
| `pyqcd-gauge` | 纯规范观测量：Wilson/Polyakov 圈、拓扑荷、Wilson flow、链接涂抹 | "Wilson 圈"/"拓扑荷"/"Wilson flow" |
| `pyqcd-analysis` | 数据分析功能链与统计方法论：02_ratio→06_FH→ana_3dir、gvar/lsqfit/SVD/色散 | "分析关联器"/"拟合数据"/"有效质量"/"ratio 图" |
| `pyqcd-tmd-chain` | 核心物理链六步：梯度流→O 组合→c0→Z_R/混合→准 TMD+NLO 匹配→连续极限 | "梯度流"/"胶子 TMD"/"Z_R"/"匹配核"/test9 |
| `pyqcd-tmd-algorithm` | 物理到实现的 TMD 算法契约：流化场强/staple→软与断连矩阵元→重整化/CS/匹配→验证门 | "TMD 算法"/"算法实现"/"staple"/"软因子"/"断连 TMD" |
| `pyqcd-pipeline` | 蒸馏管线九步运行/一致性验证/断点续跑/数据守卫/env 快照 | "跑管线"/"一致性测试"/examples/test0 |
| `pyqcd-infra` | torch 后端/h5-ASCII-VdV-VVV IO/MPI 并行显存公式 N·a=n·b | "切后端"/"h5 读写"/"MPI 并行" |
| `pyqcd-docs` | 中文 LaTeX 编译三零验收/字体规范/analy-pure 报告产物约定/报告范式 | "写文档"/"出报告"/xelatex |

## 调用链（主工作流）

```text
pyqcd-gauge ──────────────┐
                          ▼
pyqcd-physics-correlator → pyqcd-propagator        （物理推导 → 数据生产）
        │                       │
        ▼                       ▼
pyqcd-physics-spectrum    pyqcd-pipeline ◄── pyqcd-infra（后端/IO/并行底座）
        │                       │
        ▼                       ▼
   pyqcd-analysis ◄──── pyqcd-tmd-chain          （统计纪律 ← 物理链）
        │                       │
        └──────────► pyqcd-tmd-algorithm ◄── pyqcd-pipeline/infra
                                │
                                ▼
                           pyqcd-docs             （成文交验）
```

## 统一范式

各 SKILL.md 遵循同一结构：frontmatter（`name` 与目录一致 + `description`
以触发条件为主、覆盖触发词 + `metadata.openclaw.emoji`）→ 目的与边界 →
工作流程（Step 化、命令可执行）→ 常见陷阱/错误处理表 → 与其他技能配合。

执行约定：

1. **TODO**：多步骤任务执行第一步用 todowrite 工具生成详细 TODO 列表，
   每完成一个子步骤立即更新状态。
2. **证据驱动**：结论以实测为准（命令输出/退出码/数值），声明完成前逐项自检；
   引用代码路径前核实存在性——本库所有 API 路径均经 grep 核验（2026-08-25）。
3. **一次问全**：需求缺项时所有问题第一次交互一次性提出。
4. **最小改动 + 不代提交**：改动遵循仓库既有约定；git 提交由用户决定
   （或经 ~auto 预授权）。
5. **会话日志**：长任务在 cwd 生成 `.X.<时间戳>.log`（追加写、不入库），
   时间戳 `%Y-%m-%d-%H-%M-%S`。
6. **行数分级披露**：SKILL.md 目标 <500 行；超长例文拆 reference/ 目录。

## 与工作流元技能的衔接

本库为**领域技能**；任务编排/调试/优化/打标等元流程使用
/root/configure/skills 的 19 个工作流技能（~all/~debug/~optim/~up/~tag 及
`~auto-<技能>` 全自动前缀）。两库互不 import，仅按上述调用链协作。

## 维护

- 新增/修改技能须同步本注册表行与调用链图（先读后写、最小改动）；
- 上游参考来源已并入：lqcd-analysis/lqcd-physics-correlator/lqcd-physics-spectrum/
  pyquda-gauge/pyquda-tool（reference/ 例文随迁）；原目录已按用户指令删除。
