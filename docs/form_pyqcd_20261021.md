# PyQCD 2026-10-21 form 整改记录

## 目标与基线

- 仓库：`/root/PyQCD`；类型：complex；主导语言：Python，辅助 Bash、LaTeX。
- 本轮基线：`b2a6aa5`（`main` 与 `origin/main` 同点；上一轮 `dev21` 指向此提交）。
- 用户纠正上一轮将 25 个参考库分析文档登记为本地例外的裁决，要求真正整理
  `refer/git-rep`。本轮按全局 form 规则完成文档迁移与 4 个普通拷贝的 submodule 化。
- 停止条件：参考库全部为 `160000` gitlink、分析文档只在 `refer/git-rep/docs/`、
  本地审计 `unexpected=0`、全量测试通过、`git diff --check` 与大文件闸门通过。

## 审计与裁决

| 对象 | 数量/位置 | 裁决 |
|---|---|---|
| 本库参考库分析文档 | 25 个 TeX/PDF，原在 4 个库的 `docs/` | 迁移到 `refer/git-rep/docs/` |
| `EasyDistillation` | 普通拷贝 67 个跟踪文件 | 重建 submodule，固定 `f74bf7d` |
| `LQCD_Master` | 普通拷贝 5598 个跟踪文件 | 重建 submodule，固定 `e4a443e` |
| `PyQUDA` | 普通拷贝 343 个跟踪文件（含展开的 pycparser） | 重建 submodule，固定 `138d8ad`；递归初始化 `pycparser` |
| `lamet-agent` | 普通拷贝 95 个跟踪文件 | 重建 submodule，固定 `9355c52` |
| `LQCD_Skills` | 已是 submodule | 保留，补齐 README 来源 |

版本选择依据：用 `--filter=blob:none` 克隆上游，比较主仓库已跟踪文件与上游历史的
blob 集合。`LQCD_Master` 与 `lamet-agent` 找到逐文件精确匹配的 commit；
`EasyDistillation` 与 `PyQUDA` 无精确匹配，取上游 HEAD，并在下节记录差异。

### 普通拷贝与上游的差异

- `EasyDistillation`：65/67 文件相同；主仓库导入时把 `.gitattributes` 改名为
  `_gitattributes`，并使用了自定义 `.gitignore`。submodule 恢复上游文件名和
  11 个上游测试数据文件。
- `PyQUDA`：143/343 文件相同；199 个 `pyquda_core/pycparser/` 文件在上游是嵌套
  submodule（固定 `5b60167602e7914139b3e277266ccdff5c24dbf2`），主仓库此前展开为
  普通文件；`.gitattributes`/`.gitignore` 差异同上。submodule 化恢复上游嵌套结构。
- `LQCD_Master`：5598/5598 精确匹配 `e4a443e`；上游完整树 8243 个文件，额外内容为
  实验日志/数据。
- `lamet-agent`：95/95 精确匹配 `9355c52`；上游多 14 个数据/agent 文件。

## 改动内容

| 文件/路径 | 改动 |
|---|---|
| `refer/git-rep/docs/` | 新增 25 个本库分析文档（TeX/PDF），文件名不变 |
| `refer/git-rep/{EasyDistillation,LQCD_Master,PyQUDA,lamet-agent}` | 普通拷贝删除，改为 submodule |
| `.gitmodules` | 新增 4 个 submodule 条目 |
| `refer/git-rep/README.md` | path、来源、用途、固定 commit 清单，并说明 `docs/` 位置与嵌套 submodule |
| `refer/.gitignore` | 放行 `!git-rep/docs/**`，避免 `*.pdf` 挡住迁移文档 |
| `AGENTS.md` | 删除“本地例外 9”，新增参考 Git 子库规则；更新 `refer/git-rep/docs/` 产物章节 |
| `skills/form/SKILL.md` | 删除普通拷贝例外条目 |
| `skills/form/scripts/form-audit.sh` | 删除 25 条 `REFERENCE_DOC_MISPLACED` 豁免及辅助函数 |
| `refer/git-rep/docs/report_distillation_20260828.tex` | 4 处 `../lattice|example/` 改为 `../EasyDistillation/...`，修复迁移引起的相对路径断链 |

未修改 Python 生产代码、历史数据、历史日志和冻结产物；未改写已推送的 `dev21` 标签。

## 验证证据

| 检查 | 命令 | 结果 |
|---|---|---|
| Shell 语法 | `bash -n skills/form/scripts/form-audit.sh` | exit 0 |
| 本地结构审计 | `skills/form/scripts/form-audit.sh` | `registered exceptions=1147 unexpected=0` |
| 参考库索引 | `git ls-files -s -- refer/git-rep` | 5 个顶层条目均为 `160000` |
| 子模块状态 | `git submodule status --recursive` | 4 个库 + `LQCD_Skills` + 嵌套 `pycparser` 正常 |
| 全量回归 | `python -m pyqcd.testing` | `110 passed, 0 skipped, 0 failed` |
| TeX 引用 | 105 条 `\lstinputlisting` 检查 | 26 条历史 `/Users/zhangxin` 断链（迁移前已存在）；0 条迁移引入 |
| Git 空白检查 | `git diff --check` / `git diff --cached --check` | exit 0 |
| 大文件闸门 | `form-large-files-check.sh` | 脚本对 4 个 `160000` gitlink 退出码 2（已知限制）；等价检查 34 个普通暂存文件，最大 597,672 字节 < 10,000,000，通过 |

## 恢复方式

- 本轮改动整体回退：`git revert <commit>`；已推送的 `dev21` 保持不变。
- 文档迁移前位置可由 Git 历史恢复：`git show <commit>:<old-path>`。
- 4 个参考库由 `.gitmodules` 中的 URL 和 README 记录的 commit 重新初始化：
  `git submodule update --init --recursive`。

## 未决项

- `pure_*_20260815` 等历史报告仍含作者机器的 `/Users/zhangxin/...` 绝对路径；这些路径
  在迁移前即不可用，PDF 已存在，本轮不改写历史报告正文。
- `EasyDistillation` 与 `PyQUDA` 使用上游 HEAD 而非逐文件精确快照；差异已在上文记录，
  如需完全复刻旧拷贝需另找对应上游 commit。
- `PyQUDA` 的 `pyquda_core/pycparser` 是嵌套 submodule，浅克隆后必须使用
  `git submodule update --init --recursive` 初始化。
