# AGENTS.md — skills/pyqcd-momentum-smear

本目录维护 `pyqcd-momentum-smear` 技能，负责 link-phased Wuppertal、
source/sink/sequential one-smear 与相位链。

- `SKILL.md` 是权威入口，`references/` 保存动量契约。
- 方向、相位符号、源汇/顺序源角色必须显式；不得以零动量结果替代 boosted source 验证。
- 修改接口后同步 propagator 和 production-pipeline 路由。
