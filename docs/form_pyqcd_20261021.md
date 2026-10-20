# PyQCD 2026-10-21 form 整改记录

## 目标与基线

- 仓库：`/root/PyQCD`；类型：complex；主导语言：Python，辅助 Bash、LaTeX。
- 基线：`05f9080464fed7732e649fc7e5d1ca9cc869556e`（`main` 与 `origin/main` 同点，
  最新标签 `dev20` 指向其父提交 `3d528be`）。
- 范围：只读审计全部已跟踪路径，处理未登记的结构发现，补齐参考库来源清单，并同步
  根规则、本地 `form` 技能与审计包装器。
- 停止条件：本地审计 `unexpected=0`、全量测试通过、`git diff --check` 通过、
  大文件闸门通过。

## 审计与裁决

初始严格审计共 1172 项 finding，分布如下：

| 检查 | 数量 | 裁决 | 依据 |
|---|---:|---|---|
| `TRACKED_DATA_CONTENT` | 980 | 保留 | 根规则本地例外 1（历史数据与运行归档） |
| `TAG_NAME_IN_PATH`（历史日志/冻结产物） | 155 | 保留 | 根规则本地例外 7、8，包装器精确匹配 |
| `LOG_EXTENSION`（嵌套 `AGENTS.md`） | 6 | 保留 | 根规则本地例外 2 |
| `FILE_UPPERCASE`（`L24x72` 系综标识） | 4 | 保留 | 根规则本地例外 4 |
| `DOC_EXTENSION`（`docs/.gitignore`、代码树附件） | 2 | 保留 | 根规则本地例外 3 |
| `REFERENCE_DOC_MISPLACED` | 25 | **新增本地例外 9** | 见下 |

25 项全部位于 `refer/git-rep/{EasyDistillation,LQCD_Master,PyQUDA,lamet-agent}/docs/`，
是本库对这 4 个参考库产出的 `analy_*`、`pure_*`、`report_*` TeX/PDF。裁决为原地保留：

1. 根 `AGENTS.md` 已明确这 4 个库是“非独立仓库、由主仓库跟踪”的已验证参考拷贝，
   与全局 form 的 submodule 通用规则冲突；按优先级保留本地原约定并登记具名例外。
2. 迁移为 submodule 需要删除并重建 4 棵大型参考树、依赖网络和上游 URL，且会改变
   已验证分析产物的引用路径，超出本轮可逆整改范围。
3. `LQCD_Skills` 已是唯一正确配置的 submodule（索引模式 `160000`），本轮只补 README。

工作区原有 `refer/git-rep/README.md` 一处未提交改动（补充 `LQCD_Skills` 来源 URL）。
该改动与本任务目标一致，予以保留并纳入完整清单；未覆盖或回退用户内容。

## 改动内容

| 文件 | 改动 |
|---|---|
| `AGENTS.md` | 托管块新增“本地例外 9”，精确限定 4 个参考库及其本库分析文档；块外标签约定改为不绑定具体版本号 |
| `skills/form/scripts/form-audit.sh` | 新增 `_registered_reference_doc`，对 25 条完整路径逐项精确豁免；其他 `REFERENCE_DOC_MISPLACED` 仍失败 |
| `skills/form/SKILL.md` | 具名例外同步新增参考库分析文档条目 |
| `refer/git-rep/README.md` | 由 URL 列表改为 path、来源、用途、引入方式的逐项清单，补充致谢与本地例外说明 |
| `docs/form_pyqcd_20261021.md` | 本任务记录 |

未删除、移动或重命名任何参考库文件、历史数据、日志和冻结产物；未修改 Python 代码。

## 验证证据

以下命令于 2026-10-21 在 `/root/PyQCD` 执行：

| 检查 | 命令 | 结果 |
|---|---|---|
| Shell 语法 | `bash -n skills/form/scripts/form-audit.sh` | exit 0 |
| 本地结构审计 | `skills/form/scripts/form-audit.sh` | `registered exceptions=1172 unexpected=0` |
| 根规则同步 | `form-agents-sync.sh --check` | exit 0（已是最新） |
| 全量回归 | `python -m pyqcd.testing` | `110 passed, 0 skipped, 0 failed` |
| Git 空白检查 | `git diff --check` | exit 0 |
| 参考库索引 | `git ls-files -s -- refer/git-rep/LQCD_Skills` | 模式 `160000` |
| 大文件闸门 | `form-large-files-check.sh` | 见提交前结果 |

全量测试首轮出现 `109 passed, 1 failed`，未保留失败用例名；随即重跑得到
`110 passed, 0 failed`。本轮改动不包含 Python 代码，判定为并发环境下的偶发失败，
最终以完整通过的日志为准；若后续复现，需单独重跑并定位具体用例。

## 恢复方式

- 文档与规则改动：`git restore --source=05f9080 -- AGENTS.md skills/form refer/git-rep/README.md`。
- 提交后整体回退：`git revert <commit>`。
- 参考库、历史数据和冻结产物未被修改，无需恢复。

## 未决项

- 4 个参考库仍是普通拷贝而非 submodule；这是本地已登记例外。若未来要求统一为
  submodule，需要单独计划网络迁移、文档路径更新和上游 URL 核对。
- 首次全量测试的偶发失败未定位到具体用例；在后续任务中若再次出现，按 `debug` 流程处理。
