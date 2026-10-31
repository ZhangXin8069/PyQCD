# AGENTS.md — skills/pyqcd-physics-spectrum

本目录维护 `pyqcd-physics-spectrum` 技能，负责关联函数的谱分解、能量/重叠、
backward 态和拟合模板。

- `SKILL.md` 是权威入口；谱模型必须说明态截断、宇称和边界条件。
- 不承担通用数据管线或重采样实现，转交 analysis/statistics。
- 修改模型规则时同步 analysis 与 correlator 路由。
