# AGENTS.md — skills/readthedocs

本目录维护 PyQCD 的 Read the Docs 双语文档站技能：`docs/en/` 与 `docs/zh/` 的
配置、页面集合、双语代码块一致性、`-W` 构建门和推送后的重建联动。单一职责是文档站，
不发布 PyPI 版本，也不写 LaTeX 报告。

- `SKILL.md` 是权威入口，目录、命令与验收门以它为准。
- 配置文件必须叫 `.readthedocs.yaml`，且被根 `.gitignore` 放行；两份都要跟踪。
- 代码块在英文与中文之间逐字节一致；新增页同时进两份 `index.rst` 的 toctree。
- 内容视角从 PyPI 包 `pyqcd` 出发，不写仓库路径；基准命令必须标注“需源码仓库”。
- 文档改动同样走 `gpush.sh` 与 `gpush.sh -t gitee.com`。
- 修改验收门后同步 `../AGENTS.md` 的运维技能注册表与 `../README.md`。
