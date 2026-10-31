# AGENTS.md — skills/pyqcd-pipeline

本目录维护 `pyqcd-pipeline` 技能，负责运行编排、守卫、断点续跑和一致性验证。

- `SKILL.md` 是权威入口，`references/` 保存 runbook 和元数据/guard 规则。
- 不复制物理算法；每个步骤委托生产模块，失败必须关闭而非静默跳过。
- 修改入口时同步根命令、技能注册表和 `pyqcd.testing` 验证。
