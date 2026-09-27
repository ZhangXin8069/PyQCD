# `examples/` 迁移公告（2026-09-27）

仓库根目录的 `examples/` 已整体迁移到 `pyqcd/testing/`，原目录不再存在。

- 原测试/回归入口统一改为 `python -m pyqcd.testing`。
- 原 TMD、蒸馏、谱学和 donghx/lqcddb 对照代码分别进入
  `pyqcd/testing/tmd/`、`regression/` 和 `comparisons/`。
- 运行产物保留时间戳，代码目录不再使用 `dev6`、`dev7`、`test9_1`、
  `test9_2`、`cmp1` 等历史 tag 名称。
- 完整旧路径到新路径映射见 `pyqcd/testing/MIGRATION.md`。

历史日志、报告中的旧路径用于标识当时证据，不应再作为可执行路径使用。
