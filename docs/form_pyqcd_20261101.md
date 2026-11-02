# PyQCD 2026-11-01 TMD 全链交付格式审计

## 审计范围

本次任务新增或修改的 Python、测试、计划文档、运行产物说明、日志和最终报告。

## 检查结果

- `bash -n skills/form/scripts/form-audit.sh`: PASS
- `git diff --check`: PASS
- `python -B -m pyqcd.testing`: `111 passed, 1 skipped, 0 failed`
- `skills/form/scripts/form-audit.sh`: PASS，7 项已登记例外，`0 unexpected findings`。

## 既有 Finding 处理

1. `docs/absorb_dev20_gluon_tmd_audit_20261022.md`：该路径绑定历史 `dev20` 审计语境，
   已在根 `AGENTS.md` 登记精确例外，避免重命名破坏历史引用；不授权新增同类路径。
2. `refer/git-rep/README.md`：已补齐 PyQCU 来源、固定 commit 与用途，`refer/` 子库
   工作树本身未修改。

## 本次文件边界

- 新代码位于 `pyqcd/testing/tmd/` 与 `pyqcd/testing/_tmd_full_*_contract.py`。
- 生产数据位于被忽略的 `data/runs/tmd-full-20261101/`。
- 日志写入 `logs/`，最终报告写入 `docs/`。
- 只修改主仓库中的 `refer/git-rep/README.md` 治理清单；各 submodule 工作树未修改。
- 未暂存、提交、推送或创建标签。

## 恢复与未决项

12 组态生产、分析、strict verification、中文报告和日志均完成；物理状态保持
`diagnostic_only`、`physical_acceptance=false`。最终 PDF 为 29 页，三遍 XeLaTeX
均满足 Overfull、Float too large、Missing character 为零。
