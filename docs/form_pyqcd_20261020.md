# PyQCD 2026-10-20 form 整改记录

## 目标

对 `/root/PyQCD` 执行本轮 `~form`：区分真实命名冲突与已批准历史归档例外，
清理现行代码中的 tag 形标识符，移除失效外部软链接，并重新验证结构、导入和功能。

## 审计与裁决

初始严格包装器报告 `166` 个未登记 finding。逐项核对后按以下边界处理：

| 对象 | 数量/示例 | 裁决 | 依据 |
|---|---|---|---|
| `data/**` 历史数据、图像、脚本和运行归档 | 980 | 保留 | 根规则本地例外 1 |
| `docs/from-logs/**`、`logs/**`、`data/from-logs/**` | 151 个 tag 形路径 | 精确豁免 | 原日志/报告的只读历史证据 |
| `pyqcd/testing/tmd/artifacts/*/docs/analy_test9_*` | 4 个 TeX/PDF | 精确豁免 | 冻结产物，修改单侧会破坏 TeX/PDF 一致性 |
| `pyqcd/analysis/_test9_extended.py` 及 test 形函数/测试标识 | 1 个模块及关联符号 | 语义重命名 | 现行生产代码和测试标识不得使用版本标签 |
| `docs/analy_test6_20260816.tex` | 1 | 语义重命名 | 当前文档路径不应以测试版本号承担业务语义 |
| `skills/donghx/QCD3/{dev76,dev78,dev78_1,dev78_2,dev80,dev84,test12,test13,test14}` | 9 个失效软链接 | 删除 | 均指向不可用仓库外路径，且无仓库内引用 |

历史记录中的文字引用仍可保留，用于说明版本来源；本次未删除或迁移任何历史数据。

## 改动内容

- 将 `pyqcd/analysis/_test9_extended.py` 改名为 `_extended_plots.py`。
- 将 `generate_test0_style_plots`、`generate_test6_style_plots` 和
  `_load_test9_corr2_raw` 改为按职责命名的
  `generate_correlator_style_plots`、`generate_energy_style_plots` 与
  `_load_corr2_raw`。
- 同步更新 `pyqcd.analysis` 导出、TMD 扩展图表入口、三个契约测试模块及
  `pyqcd.testing` 注册表；测试类名和测试方法名不再包含版本标签。
- 将 `docs/analy_test6_20260816.tex` 改名为
  `docs/analy_proton_energy_20260816.tex`。
- 删除 9 个无引用且已失效的外部标签形软链接。
- 将根 `AGENTS.md` 的 form 规则改为唯一托管标记块，并增加历史归档与冻结产物例外。
- 更新 `skills/form/SKILL.md` 与 `skills/form/scripts/form-audit.sh`，对新增例外使用
  精确路径匹配，其他 finding 仍失败。

## 验证证据

以下命令均于 2026-10-20 在 `/root/PyQCD` 执行：

| 检查 | 命令 | 结果 |
|---|---|---|
| 改动前全量基线 | `python -m pyqcd.testing` | `110 passed, 0 skipped, 0 failed` |
| 受影响的契约测试 | `python -m unittest pyqcd.testing._analysis_safety_contract pyqcd.testing._fit_status_propagation_contract pyqcd.testing._statistics_edge_contract` | `59` 项通过 |
| 扩展图表入口 | `python -m pyqcd.testing.tmd.extended_plots --help` | exit 0 |
| 全量回归 | `python -m pyqcd.testing` | `110 passed, 0 skipped, 0 failed` |
| 严格结构审计 | `skills/form/scripts/form-audit.sh` | `registered exceptions=1148 unexpected=0` |
| Python 编译 | `python -m compileall -q pyqcd` | exit 0 |
| Git 空白检查 | `git diff --check` | exit 0 |
| 标签历史链 | `tag-chain.sh --check --remote origin` | 88 个标签全部通过 |

全量测试中的协方差非正定警告来自既有回退路径，测试按预期通过；本轮未修改该数值逻辑。

## 恢复方式

- 软链接删除和文件重命名均保存在 Git 历史中；需要恢复时使用
  `git restore --source=HEAD -- <path>`，提交后使用 `git revert <commit>`。
- 历史归档没有被删除；如需读取旧版本，使用对应 commit 或 tag。
- 回退本次提交不会改变远端标签或历史数据。

## 未决项

- 无代码功能未决项。
- 历史 `docs/from-logs/**`、`logs/**`、`data/from-logs/**` 的标签形目录名继续作为
  版本证据保留；后续新增路径不得复用该例外。
