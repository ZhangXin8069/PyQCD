# AGENTS.md — skills/pyqcd-tmd-algorithm

本目录维护 `pyqcd-tmd-algorithm` 技能，负责 TMD 几何、重整化、匹配、外推和
物理验证门。

- `SKILL.md` 是权威入口；`references/` 与 `evals/` 保存算法细节和证据案例。
- 固定流时、单系综或缺少 soft/rapidity 时只能标记 `diagnostic_only`。
- 修改验收门后同步 tmd-chain、pipeline、docs 与相关测试。
