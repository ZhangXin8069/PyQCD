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
## form-20261021-refer - final - 2026-10-21 04:16:14 +0800
- 主任务: PyQCD ~form：迁移参考库分析文档并把 refer/git-rep 重建为 submodule
- 日志文件: logs/v202610210416.txt
- 文件大小: 5633 bytes
- 触发原因: 任务完成
- 关键内容: 25 个分析文档迁入 refer/git-rep/docs，4 个普通拷贝改为 submodule，审计 0 未登记，110 项回归通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## agent-config-tui-fix - final - 2026-10-21 22:54:55 +0800
- 主任务: debug co go 启动失败
- 日志文件: logs/v2026102122.txt
- 文件大小: 3230 bytes
- 触发原因: 任务完成
- 关键内容: 定位并修复 co go TUI 启动漏传 CODEX_BIN 的回归
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## form-audit-20261022 - final - 2026-10-22 01:34:44 +0800
- 主任务: PyQCD form 只读格式审计（main，无整改项）
- 日志文件: logs/v2026102201.txt
- 文件大小: 4504 bytes
- 触发原因: 任务完成
- 关键内容: form 审计通过：registered exceptions=1147 unexpected=0；110 passed/0 failed
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## dev20-absorb-debug - final - 2026-10-22 03:09:58 +0800
- 主任务: dev20 外来内容吸收与胶子 TMD-PDF 全面 debug
- 日志文件: logs/v2026102203.txt
- 文件大小: 7011 bytes
- 触发原因: 任务完成
- 关键内容: 111 tests pass; source-resolved real conf6250 smoke; physical PDF remains blocked by missing soft/mixing/multi-scale data
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## dev20-absorb-debug-2 - final - 2026-10-22 06:51:46 +0800
- 主任务: dev20 外来内容吸收与胶子 TMD-PDF 全面 debug
- 日志文件: logs/v2026102206.txt
- 文件大小: 11485 bytes
- 触发原因: 任务完成
- 关键内容: 修复 C2×OPE ensemble 归一化、replica AIC、helicity 复数分母/Fourier/soft 域；112 项回归通过；物理 PDF 仍因 P000 质量和多尺度缺失而 blocked
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## dev20-absorb-debug-3 - final - 2026-10-22 07:18:08 +0800
- 主任务: dev20 外来内容吸收与胶子 TMD-PDF 全面 debug
- 日志文件: logs/v2026102207.txt
- 文件大小: 8388 bytes
- 触发原因: 任务完成
- 关键内容: 扩展到 11 组态并采用 1000 bootstrap；记录 flow b-window 失败；112 项回归通过，物理 PDF 仍 blocked
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## dev20-absorb-debug-4 - final - 2026-10-22 07:22:06 +0800
- 主任务: dev20 外来内容吸收与胶子 TMD-PDF 全面 debug
- 日志文件: logs/v2026102207-2.txt
- 文件大小: 8544 bytes
- 触发原因: 任务完成
- 关键内容: 完成 11 组态/1000 bootstrap 真实诊断、严格 LIME 解析、flow b-window fail-closed；112 项回归通过，物理 PDF 仍 blocked
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## dev20-absorb-debug-5 - final - 2026-10-22 07:55:38 +0800
- 主任务: dev20 外来内容吸收与胶子 TMD-PDF 全面 debug
- 日志文件: logs/v2026102207-3.txt
- 文件大小: 7481 bytes
- 触发原因: 任务完成
- 关键内容: 完成 11 组态 1000 bootstrap、tau=0.5/1.0 受控 flow-window 扫描、严格 LIME I/O；112 项回归通过，物理 PDF 仍 blocked
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

