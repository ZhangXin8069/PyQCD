# 共享 Skill 目录

本文件是当前目录中所有 Codex skills 的权威目录。Agent 应先根据“功能”字段选择与当前任务匹配的最小 skill 集合，再按需读取所选 skill；禁止一次性加载全部 skill。

## Skill 列表

| Skill 目录 | 创建者 | 功能 | 最近一次更改时间 | 入口 |
| --- | --- | --- | --- | --- |
| [`lqcddb/`](lqcddb/) | `sush` | 审查和使用 lqcddb 格点 QCD distillation 和 blending 包，涵盖 Wick 收缩、perambulator、本征矢顶角、关联函数构造、MPI、统计分析、有效质量、GEVP、动量换算及收缩性能。适用于强子关联函数、distillation、blending、格点谱学及 lqcddb API 或源码审查。 | `2026-08-27 17:47:59 +08:00` | [`SKILL.md`](lqcddb/SKILL.md) |

## 目录维护规则

- 每个 skill 必须有且仅有一条记录，并且其目录内必须存在 `SKILL.md`。
- 新建、更新、重命名或删除 skill 时，必须在同一任务中同步更新本表。
- “创建者”必须是 skill 目录的操作系统所有者；已有记录的创建者不得由 agent 修改。
- “功能”应简洁描述适用任务和触发范围，使 agent 无需读取全部 skill 即可完成初筛。
- “最近一次更改时间”采用 `YYYY-MM-DD HH:MM:SS +08:00` 格式；skill 内任何内容或名称变化后都必须更新。
- Agent 只能修改当前操作系统用户所拥有 skill 的记录，不得改动其他创建者的记录。

完整的所有权核验、按需加载、提交与验证规则见 [`AGENTS.md`](AGENTS.md)。
