# PyQCD form 格式治理（2026-09-28）

## 目标与基线

- 仓库：`/root/PyQCD`
- 类型：complex
- 主导语言：Python；辅助语言：Bash、LaTeX
- 整改前基线：`811900c76c5b9fb3632ee855a9c2ae97e6df76b6`（`main`、`origin/main`、
  `dev15` 同点）
- 目标：把命名、顶层目录、归档边界、本地审计和 Git 交付规则固化为可执行文档，不改动
  物理实现、历史数据和只读参考内容。

## 审计证据

全局 `form-audit.sh` 对跟踪文件报告 994 项后发现：

| 规则 | 数量 | 处理 |
|---|---:|---|
| `TRACKED_DATA_CONTENT` | 981 | `data/**` 历史数据与运行归档，登记为具名例外 |
| `LOG_EXTENSION` | 7 | `logs/**/AGENTS.md` 目录治理文件，登记为具名例外 |
| `FILE_UPPERCASE` | 4 | 冻结基线 dataset slug 中的 `L24x72`，保留外部接口名称 |
| `DOC_EXTENSION` | 2 | 文档局部 `.gitignore` 与报告 `lstinputlisting` 附件 |

审计未发现未登记顶层目录、测试位置错误、临时备份文件或空白文件名。另确认：

- `cpp/` 已在提交 `228fb47` 删除，当前根 `AGENTS.md` 不再把它声明为现行目录。
- `.opencode/skills/` 当前不存在，目录说明改为实际的 `skills/` 树。
- 顶层空目录 `"docs` 不含任何文件，已移除该命名错误；需要时可执行
  `mkdir -- '"docs'` 恢复空目录。
- 基线 `811900c` 中 60 余个数学符号或上游兼容 API 名保留原拼写，不进行破坏性改名。

## 改动

| 文件 | 改动 |
|---|---|
| `AGENTS.md` | 新增“form 格式约定”，登记目录、命名、验证、Git 和六类本地例外 |
| `skills/form/SKILL.md` | 新增 PyQCD 本地 form 运维技能 |
| `skills/form/scripts/form-audit.sh` | 新增具名例外审计包装器，任何新发现失败 |
| `skills/AGENTS.md` | 登记 form 运维技能并明确领域/运维边界 |
| `skills/README.md` | 增加运维技能发现入口 |
| `.gitignore` | 放行 `skills/form/scripts/**`，修正归档树忽略检查说明 |
| `docs/AGENTS.md` | 登记本格式治理任务文档 |

未移动或删除任何 `data/**`、`refer/**`、冻结基线内容或历史报告。

## 验证

```text
bash -n skills/form/scripts/form-audit.sh
  PASS

python /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/form
  Skill is valid!

skills/form/scripts/form-audit.sh
  PASS form-audit(pyqcd): registered exceptions=994 unexpected=0

git ls-files -ci --exclude-standard -- . ':!refer/**' ':!data/**'
  0 lines

python -m pyqcd.testing
  104 passed, 2 skipped, 0 failed
```

整改前全量测试曾出现一次 MPI collective 用例 `rc=124`；该用例单独重跑通过，整改后
全量测试也通过，判定为并发环境下的偶发超时。跳过项是 `gvar` 未安装和
`physics.sty` 缺失，均有显式 skip，不是静默放行。

## 恢复

格式文件可从基线恢复：

```bash
git restore --source=811900c -- AGENTS.md .gitignore skills/
```

任务文档和空目录清理不影响可复现数据；历史数据若需回到更早版本，按 `git log --follow`
定位原对象，不执行目录级破坏性恢复。

## 后续

新增合法例外时，必须同时更新根 `AGENTS.md`、本地 `form` 技能和审计包装器的精确路径，
再运行全量验证。环境依赖缺口 `gvar`、`physics.sty` 留给对应功能任务处理，不属于本次
格式治理范围。
