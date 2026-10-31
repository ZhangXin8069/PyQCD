# AGENTS.md — skills/pyqcd-hisq-recover

本目录维护 `pyqcd-hisq-recover` 技能，负责 staggered/HISQ 到 naive 四旋量存储的
spin lift。

- `SKILL.md` 是权威入口，`references/` 保存 producer contract。
- 明确 full-grid、MPI parity 和自旋恢复边界；不得静默改轴或奇偶布局。
- 修改后运行相关结构性检查并同步 README。
