# AGENTS.md — skills/pyqcd-tmd-chain

本目录维护 `pyqcd-tmd-chain` 技能，负责梯度流胶子 TMD-PDF 的端到端导航、
阶段关系与状态判断。

- `SKILL.md` 是权威入口；实现细节路由到 `pyqcd-tmd-algorithm`。
- 明确当前阶段、输入尺度和 `physical_acceptance`，不得把导航清单当作通过证据。
- 修改调用关系时同步技能注册表、调用链和 README。
