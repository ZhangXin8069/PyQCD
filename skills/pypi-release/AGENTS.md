# AGENTS.md — skills/pypi-release

本目录维护 `pyqcd` 的 PyPI 发布运维技能：三处版本号同步、sdist 构建与内容校验、
twine 上传、CDN 传播等待、干净 venv 安装自检和回滚策略。单一职责是发布流程，
不判断物理正确性。

- `SKILL.md` 是权威入口，命令、检查项与验收门以它为准。
- 只发 sdist（`pyqcd-<ver>.tar.gz`），不上传 wheel；分发名和导入名都是 `pyqcd`。
- 凭据只从 `~/env-key.sh` 的来源环境读 `PYPI_ZHANGXIN8069_API_KEY`，禁止写入文件或打印。
- 默认自动发版；回滚、删包、改名、降级必须先问用户。
- 每次发布后必须执行 `gpush.sh` 与 `gpush.sh -t gitee.com`。
- 修改验收门后同步 `../AGENTS.md` 的运维技能注册表与 `../README.md`。
