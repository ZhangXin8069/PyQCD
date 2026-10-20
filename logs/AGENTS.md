# AGENTS.md — logs

仅存放日志与运行记录（如 `txt/json/log/tsv`）；图像、数据、脚本和安装文件统一归档于
`data/from-logs/`，文档统一归档于 `docs/from-logs/`。嵌套 `AGENTS.md` 作为目录治理例外保留。

## 批次索引

### 20261020-git-slimming - final - 2026-10-20 01:03:46 +0800
- 主任务: Git 历史大文件与 LFS 清理
- 日志文件: logs/v2026102001.txt
- 文件大小: 5962 bytes
- 触发原因: 任务完成
- 关键内容: 删除12个大文件及全部LFS历史内容，重写并同步main与88个标签；Gitee服务端GC待网页触发
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

### 20261020-git-slimming-gc-verify - final - 2026-10-20 01:08:39 +0800
- 主任务: Gitee GC 后远程物理删除复核
- 日志文件: logs/v2026102001-2.txt
- 文件大小: 3510 bytes
- 触发原因: 任务完成
- 关键内容: Gitee GC后旧main SHA仍可读取，逻辑删除完成但服务端物理回收未确认
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

### pyqcd-form-20261020 - final - 2026-10-20 01:21:15 +0800
- 主任务: PyQCD 2026-10-20 form 整改与功能复现
- 日志文件: logs/v2026102001-3.txt
- 文件大小: 5660 bytes
- 触发原因: 任务完成
- 关键内容: 清理现行标签形代码标识符，登记历史归档例外，完整回归 110 项通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## reference-review-20261020 - final - 2026-10-20 03:27:57 +0800
- 主任务: 审查新增参考文档与代码并合并修正
- 日志文件: logs/v2026102003.txt
- 文件大小: 6063 bytes
- 触发原因: 任务完成
- 关键内容: 修复 TMD ratio、NLO matching、Fourier 归一化及 fail-closed 数值门；全量 110 项通过；一致性验证因缺失大体积参考产物未执行
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## form-20261021 - final - 2026-10-21 04:01:36 +0800
- 主任务: PyQCD ~form 格式治理：登记参考库分析文档本地例外并补齐来源清单
- 日志文件: logs/v2026102104.txt
- 文件大小: 5037 bytes
- 触发原因: 任务完成
- 关键内容: 25 项 REFERENCE_DOC_MISPLACED 登记为本地例外 9，审计 0 未登记，全量测试 110 passed
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
