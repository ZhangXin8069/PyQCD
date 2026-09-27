---
name: form
description: |
  Use when auditing or maintaining PyQCD repository naming, directory layout,
  documentation/log/data placement, delivery format, local exceptions, or form
  validation.
metadata:
  openclaw:
    emoji: 📐
---

# form — PyQCD 格式治理

## 目的与边界

本技能维护 `/root/PyQCD` 的命名、顶层目录、测试归集、归档边界、文档交付和 Git
验收格式。它不负责物理正确性、数值算法优化或修复普通运行错误；发现这些问题时转交
对应的 `pyqcd-*` 领域技能。

根 `AGENTS.md` 的“form 格式约定”是本地权威规则。全局 `form` 技能提供通用审计流程
和参考，不能据此删除本项目已登记的历史归档。

## 本地判定

- 本库为 complex 库，生产语言以 Python 为主，Bash 和 LaTeX 为辅。
- 生产包为 `pyqcd/`，测试统一归入 `pyqcd/testing/`。
- 顶层目录白名单为 `pyqcd/`、`data/`、`docs/`、`logs/`、`refer/`、`skills/`。
- Python 模块使用小写下划线；私有模块和符号前缀 `_`；类型使用大驼峰。
- 基线 `811900c` 的数学符号名和上游兼容入口保留原拼写；新普通接口仍用
  `snake_case`。
- LaTeX/Markdown 使用内容或领域名称；日志仅允许 `log/json/tsv/csv/txt`。
- `refer/` 永远只读；生产代码不得依赖 `refer/` 或 `pyqcd.testing`。
- `data/**` 不是空数据目录，而是本库批准的历史数据、图像、脚本和运行归档树。

## 工作流程

1. 从 Git 根目录开始，确认分支、工作区、远端和最新标签；不得扫描仓库外内容。
2. 运行 `skills/form/scripts/form-audit.sh`。包装器调用全局审计，并只过滤根
   `AGENTS.md` 已登记的例外。
3. 对每个新发现给出 `文件:行号` 或路径证据，区分真实冲突、兼容例外和归档例外。
   **不得以“历史文件”为由跳过未登记的新违规。**
4. 把移动、引用更新和验证放入同一批次；禁止机械小写化数学量、外部系综标识和
   上游只读文件。
5. 改动后运行全量测试、相关模块冒烟、`bash -n`、`git diff --check`，并再次运行本地
   form 审计。
6. 在 `docs/form_pyqcd_<YYYYMMDD>.md` 记录冲突摘要、改动、证据、恢复方法和未决项。
7. 仅暂存本任务文件，普通提交和推送；使用下一个 `dev<N>` 标签保存验收快照。

## 具名例外

- `data/**`：允许历史数据、图像、脚本和说明文件，按已验证运行归档保留。
- `logs/**/AGENTS.md`：允许嵌套治理文件与文本日志共存。
- `docs/.gitignore`：文档局部忽略元数据。
- `docs/_tree_pyqcd.txt`：`analy_pyqcd_20260818.tex` 的 `\lstinputlisting` 附件。
- 基线 `811900c` 中承载数学含义或上游兼容性的 Python 符号保留原拼写。
- 冻结基线中的 dataset slug 脚本名保留 `L24x72` 等物理标识，不把文件名小写化。
- `refer/**` 与 `data/**` 中已跟踪且命中局部忽略规则的历史资料保持原状。

除上述条件外，新增例外必须先写入根 `AGENTS.md` 并在审计包装器中按精确路径登记。

## 验证门

```bash
bash -n skills/form/scripts/form-audit.sh
skills/form/scripts/form-audit.sh
python -m pyqcd.testing
git diff --check
```

审计包装器必须以 0 退出，并报告预期例外数及 `0 unexpected findings`。全量测试中的
环境依赖跳过或 MPI 超时需要独立重跑；未复现的功能失败必须明确记录，不能改期望值掩盖。

## 错误处理

- 工作区已有用户改动时只记录，不覆盖；无法安全分离时不提交。
- 新发现命中历史归档时，先查 `git log --follow --diff-filter=A` 和引用，再决定是否
  登记例外；不得直接删除。
- 本地包装器出现未预期条目时停止提交，修正路径或代码后重跑。
- 推送非快进失败时先取回并分析，禁止 force push；标签漂移交给 `tag` 处理。

## 交接

- 物理、数值或管线规则分别交给对应 `pyqcd-*` 技能。
- 旧引用排查交给 `diff`，运行失败交给 `debug`，功能回归交给 `test`。
- 完成后把规则同步到根 `AGENTS.md`、`skills/README.md`、本技能和任务文档。
