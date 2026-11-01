# PyQCD 2026-11-01 TMD 全链交付格式审计

## 审计范围

本次任务新增或修改的 Python、测试、计划文档、运行产物说明、日志和最终报告。

## 检查结果

- `bash -n skills/form/scripts/form-audit.sh`: PASS
- `git diff --check`: PASS
- `python -B -m pyqcd.testing`: `111 passed, 1 skipped, 0 failed`
- `skills/form/scripts/form-audit.sh`: 退出 1，报告 2 项既有 finding，均不属于本次新增文件。

## 既有 Finding

1. `docs/absorb_dev20_gluon_tmd_audit_20261022.md`：路径包含 `dev20` tag 形式名称。该文件为既有历史审计文档，本任务未修改；不做重命名，以免破坏历史引用。
2. `refer/git-rep/README.md`：未记录 submodule `PyQCU` 的来源。该文件属于外来参考库治理范围，本任务未修改；不擅自改写外部参考元数据。

## 本次文件边界

- 新代码位于 `pyqcd/testing/tmd/` 与 `pyqcd/testing/_tmd_full_*_contract.py`。
- 生产数据位于被忽略的 `data/runs/tmd-full-20261101/`。
- 日志写入 `logs/`，最终报告写入 `docs/`。
- 未修改 `refer/` 子库工作树，未暂存、提交、推送或创建标签。

## 恢复与未决项

全量物理生产仍需在用户明确授权存储策略后从 worker checkpoint 继续。当前磁盘实际占用约 77.9%，现有缓存和 checkpoint 均保留。
