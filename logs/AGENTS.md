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

## dev24 - final - 2026-10-22 13:58:04 +0800
- 主任务: 创建 dev24 Git 开发快照标签
- 日志文件: logs/v2026102213.txt
- 文件大小: 5144 bytes
- 触发原因: 任务完成
- 关键内容: 验证 dev23..HEAD 后创建本地 annotated tag dev24，全量回归 112 passed，未推送远程
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## tag-dev - final - 2026-10-24 21:29:37 +0800
- 主任务: 创建 dev26 本地注释标签
- 日志文件: logs/v2026102421.txt
- 文件大小: 4676 bytes
- 触发原因: 任务完成
- 关键内容: dev26 已创建并验证，标签链通过；文档登记与尾随空白待后续处理，未推送远程
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## tmd-equivalence - final - 2026-10-24 22:01:09 +0800
- 主任务: PyQCD 与 refer/donghx 的核子胶子 TMD-PDF 等价性审计
- 日志文件: logs/v2026102422.txt
- 文件大小: 9468 bytes
- 触发原因: 任务完成
- 关键内容: 结论：整库及完整核子胶子 TMD-PDF 不等价；直线胶子 OPE、flow-to-quasi 与 gg 匹配核为算法/移植等价，PyQCD 的横向 staple TMD 是扩展诊断原型。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## donghx-cross-reference - final - 2026-10-24 22:20:55 +0800
- 主任务: PyQCD 与 refer/donghx 在 4x4x4x8 格点上的胶子 OPE 对照测试
- 日志文件: logs/v2026102422-2.txt
- 文件大小: 8720 bytes
- 触发原因: 任务完成
- 关键内容: 同一 4x4x4x8 SU(3) thin-link 场上完成双方实跑；共享 flow/OPE/matching 链 23/23 通过并两次复现，完整横向 TMD 因 donghx 无对应实现不可直接对照。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## handoff-donghx-cross-reference - final - 2026-10-24 22:26:59 +0800
- 主任务: 4x4x4x8 PyQCD/donghx 对照测试跨设备交接
- 日志文件: logs/v2026102422-3.txt
- 文件大小: 41261 bytes
- 触发原因: 任务完成
- 关键内容: 单一日志包含完整源码补丁、环境、哈希、23/23 对照结果、回归失败清单与跨设备恢复步骤。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## cross-device-resume - final - 2026-10-24 22:33:41 +0800
- 主任务: 衔接 4x4x4x8 PyQCD/donghx 对照测试跨设备交接
- 日志文件: logs/v2026102422-4.txt
- 文件大小: 5266 bytes
- 触发原因: 任务完成
- 关键内容: Linux 本机复跑 23/23 PASS，固定输入 SHA 匹配，生成并校验 artifact。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## tmd-finite-b-oracle - final - 2026-10-24 22:57:21 +0800
- 主任务: 扩展有限 b_perp staple/TMD 验证
- 日志文件: logs/v2026102422-5.txt
- 文件大小: 5126 bytes
- 触发原因: 任务完成
- 关键内容: 新增常量链接闭式路径、b=0/L 无关性和纯规范零场强三项 oracle；全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-ratio-v2 - final - 2026-10-24 23:08:51 +0800
- 主任务: 有限 b_perp 胶子 TMD C3/C2 ratio-v2 长任务
- 日志文件: logs/v2026102423.txt
- 文件大小: 6889 bytes
- 触发原因: 任务完成
- 关键内容: 新增有限 b ratio-v2 公共 API、显式 b 元数据、独立 jackknife oracle 与中心-only JK 状态；全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-integration-gate - final - 2026-10-24 23:33:45 +0800
- 主任务: 有限 b_perp TMD 端到端输入审计与跨组态统计集成
- 日志文件: logs/v2026102423-2.txt
- 文件大小: 8095 bytes
- 触发原因: 任务完成
- 关键内容: 新增 source-resolved 跨组态 ratio-v2 API 与只读输入审计 CLI；真实工作区因 Nconf=1、C2 非源分辨且 OPE 非全通道而 blocked；全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-full-channel-ope - final - 2026-10-25 00:02:53 +0800
- 主任务: 有限 b_perp 全通道 TMD OPE producer 与 canonical cache
- 日志文件: logs/v2026102500.txt
- 文件大小: 7115 bytes
- 触发原因: 任务完成
- 关键内容: 新增 (2,4,6,nz,nb,Nt) producer、独立 full-channel v2 cache/loader 与严格审计；b=0 全方向 straight 差异 1.243e-14；全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-end-to-end-synthetic - final - 2026-10-25 00:06:11 +0800
- 主任务: 两配置有限 b TMD audit-to-ratio 合成端到端链
- 日志文件: logs/v2026102500-2.txt
- 文件大小: 5580 bytes
- 触发原因: 任务完成
- 关键内容: 完成 producer→OPE cache→source-resolved C2 cache→audit ready→ratio-v2→Im(R_H) 两配置受控闭环；物理契约 16/16、全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-2d-geometry - final - 2026-10-25 00:12:41 +0800
- 主任务: 二维有限横向位移 staircase 与 full-channel TMD OPE geometry
- 日志文件: logs/v2026102500-3.txt
- 文件大小: 5002 bytes
- 触发原因: 任务完成
- 关键内容: 新增显式 transverse_order 的二维 staple、2D 端点矩阵元与 (2,4,6,nz,nbx,nby,Nt) producer；常量闭式、规范协变、1D 退化均通过；全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete

## finite-b-2d-cache - final - 2026-10-25 00:21:42 +0800
- 主任务: 二维 full-channel TMD OPE canonical cache 与 loader
- 日志文件: logs/v2026102500-4.txt
- 文件大小: 5361 bytes
- 触发原因: 任务完成
- 关键内容: 新增 v3 2D cache contract/compute/loader，绑定 bx/by 网格、方向与 transverse_order；三请求隔离、1D loader 不误命中、全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## finite-b-ratio-v3-public-data - final - 2026-10-25 01:46:42 +0800
- 主任务: 二维 finite-b ratio-v3、真实 full-channel OPE 与跨组态审计
- 日志文件: logs/v2026102501.txt
- 文件大小: 7752 bytes
- 触发原因: 任务完成
- 关键内容: 完成 ratio-v3/cross-config、full-channel OPE/cache、真实 4150/6250 输入审计；全量 112/112 PASS，结构 ready，真实 pol35 仍缺失。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## polarized-c2-v3-real-4150-6250 - final - 2026-10-25 02:49:07 +0800
- 主任务: source-resolved nopol/pol35 v3 C2 与真实 4150/6250 ratio 接线
- 日志文件: logs/v2026102502.txt
- 文件大小: 6326 bytes
- 触发原因: 任务完成
- 关键内容: 完成 v3 双极化合同、真实 4150/6250 C2、audit v2 和 ratio 接线；4150 公共参考误差 1e-8/1e-6，全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## readonly-size-audit - final - 2026-10-25 03:39:49 +0800
- 主任务: 统计 Gradient Flow GluonPDF 参考目录数据大小
- 日志文件: logs/v2026102503.txt
- 文件大小: 2951 bytes
- 触发原因: 任务完成
- 关键内容: 统计当前目录逻辑与实际占用大小，交叉核验文件数量且不跟随符号链接
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## key-data-size-audit - final - 2026-10-25 05:06:34 +0800
- 主任务: 统计 Gradient Flow GluonPDF 相关关键数据大小
- 日志文件: logs/v2026102505.txt
- 文件大小: 4289 bytes
- 触发原因: 任务完成
- 关键内容: 统计当前代码及两套组态、传播子、本征矢、2pt、流动结果与OPE派生数据；记录权限缺口
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## data-list-v20261025 - final - 2026-10-25 05:21:14 +0800
- 主任务: 生成 Gradient Flow GluonPDF 数据路径与意义清单
- 日志文件: logs/v2026102505-2.txt
- 文件大小: 3118 bytes
- 触发原因: 任务完成
- 关键内容: 创建 222 行数据路径清单，修正并核验总量与权限缺口
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## data4pyqcd-2tb - final - 2026-10-25 05:39:29 +0800
- 主任务: 生成 2TB 预算内的 PyQCD 数据取样下载清单
- 日志文件: logs/v2026102505-3.txt
- 文件大小: 3129 bytes
- 触发原因: 任务完成
- 关键内容: 101 条分层取样路径，含 4 个传播子配置和 33 个 donghx 对照结果
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## data4pyqcd-downloader - final - 2026-10-25 05:45:16 +0800
- 主任务: 编写 data4pyqcd 清单远程下载脚本
- 日志文件: logs/v2026102505-4.txt
- 文件大小: 2927 bytes
- 触发原因: 任务完成
- 关键内容: 实现清单驱动、dry-run、断点续传和双远端根目录的 SSH 下载脚本
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## pyqcd-infra-data-transfer - final - 2026-10-25 05:49:39 +0800
- 主任务: 将对照数据取样经验写入 pyqcd-infra 技能
- 日志文件: logs/v2026102505-5.txt
- 文件大小: 3475 bytes
- 触发原因: 任务完成
- 关键内容: 新增 data-transfer reference 并同步技能入口、注册表和验证
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## all11-polarized-tmd-diagnostic - final - 2026-10-25 08:24:36 +0800
- 主任务: 11 组态极化 gluon TMD 真实诊断链与 GPU flow 内存修复
- 日志文件: logs/v2026102508.txt
- 文件大小: 7357 bytes
- 触发原因: 任务完成
- 关键内容: 完成 11 组态 v3 C2、1D/2D full-channel OPE、audit、ratio、JK 协方差与 staircase 对照；修复 GPU flow OOM，全量 112/112 PASS。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## dev25-report-20261025 - final - 2026-10-25 18:10:40 +0800
- 主任务: 恢复 dev25 胶子 TMD 报告任务：PDF 交付与 112 项全量回归验证
- 日志文件: logs/v2026102518.txt
- 文件大小: 5833 bytes
- 触发原因: 任务完成
- 关键内容: 交付 30 页 PDF（三零）；112/112 回归通过；未提交
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## dev25-report-20261025 - general - 2026-10-25 18:15:13 +0800
- 主任务: dev25 报告收尾：审计问题闭环复核与文档索引同步
- 日志文件: logs/v20261025181513.txt
- 文件大小: 2521 bytes
- 触发原因: 一般记录
- 关键内容: 审计高优先级项 0 遗留；AGENTS.md 登记 dev25 报告
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## tmd-audit-20261025 - final - 2026-10-25 19:13:08 +0800
- 主任务: dev25 报告后续：分母约定量化、τ=1 流窗口试点与入库
- 日志文件: logs/v2026102519.txt
- 文件大小: 5797 bytes
- 触发原因: 任务完成
- 关键内容: 4 提交；分母差 5.7σ；τ=1 试点 11 组态；33 页 11 图三零；112/112
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## v20261026 - final - 2026-10-26 03:40:41 +0800
- 主任务: rename and fix download_data4pyqcd-v20261026.sh
- 日志文件: logs/v2026102603.txt
- 文件大小: 5392 bytes
- 触发原因: 任务完成
- 关键内容: Fix SCP directory transfers and preserve remote absolute paths from /
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## v20261026 - final - 2026-10-26 03:51:29 +0800
- 主任务: fix download skip-existing re-download behavior
- 日志文件: logs/v2026102603-2.txt
- 文件大小: 6764 bytes
- 触发原因: 任务完成
- 关键内容: Verify unmarked existing files by remote/local size and SHA-256 before skipping
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## v20261026 - final - 2026-10-27 00:13:53 +0800
- 主任务: 新增 L24x72 50 组态完整数据下载脚本
- 日志文件: logs/v2026102700.txt
- 文件大小: 6763 bytes
- 触发原因: 任务完成
- 关键内容: 新增 50 组态、1200 项数据传输脚本，完成 dry-run、清单与静态验证；远端认证未通过
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## v20261026 - final - 2026-10-27 00:24:08 +0800
- 主任务: 收紧 L24x72 50 组态下载脚本为 core-only
- 日志文件: logs/v2026102700-2.txt
- 文件大小: 5566 bytes
- 触发原因: 任务完成
- 关键内容: 默认改为 gauge/eigensystem/light perambulator 共 150 项；增强数据改为显式开关；补充 scp+tar 目录回退
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## donghx-gradient-flow-audit-20261027 - stage - 2026-10-27 01:35:42 +0800
- 主任务: Donghx Gradient Flow GluonPDF 全面 debug 与小实例比对
- 日志文件: logs/v202610270135.txt
- 文件大小: 12638 bytes
- 触发原因: 阶段成果
- 关键内容: C++/PyQCD 小实例对照与归档树缺陷审计完成，真实数据子代理待完成
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## resume-20261027T003813.r7UP9v - final - 2026-10-27 05:57:04 +0800
- 主任务: Donghx 梯度流胶子 PDF 审计与真实数据 TMD-PDF 失败关闭诊断
- 日志文件: logs/v2026102705.txt
- 文件大小: 8518 bytes
- 触发原因: 任务完成
- 关键内容: 恢复任务完成：23/23 跨实现验证、34/34 独立验证、27 图与报告生成，物理结论保持 diagnostic_only
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## download-scripts-20261028 - final - 2026-10-28 20:28:44 +0800
- 主任务: 为 download_**-v20261026.sh 添加多线程下载
- 日志文件: logs/v2026102820.txt
- 文件大小: 3657 bytes
- 触发原因: 任务完成
- 关键内容: 两个下载脚本新增 --jobs/-j 并发传输、并发安全计数与失败日志
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## 20261028T21-resume - final - 2026-10-29 02:18:46 +0800
- 主任务: 恢复 20261028T21 系列 TMD 梯度流运行
- 日志文件: logs/v2026102902.txt
- 文件大小: 6870 bytes
- 触发原因: 任务完成
- 关键内容: 完成五个组态续跑、9 组态分析、零检验、复算验证、图件与报告；保持 diagnostic_only
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## resume-20261029T010609.rMWf2s - general - 2026-10-29 03:51:32 +0800
- 主任务: 整理用户输入指令并归档
- 日志文件: logs/v20261029035132.txt
- 文件大小: 577 bytes
- 触发原因: 一般记录
- 关键内容: 身份规则、恢复目标、资源约束与日志归档要求的一句话摘要
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## dev29-to-20261029 - general - 2026-10-29 03:56:22 +0800
- 主任务: 整理 dev29 以来用户实质输入指令
- 日志文件: logs/v20261029035622.txt
- 文件大小: 756 bytes
- 触发原因: 一般记录
- 关键内容: dev29 以来用户实质输入指令的一句话汇总
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## resume-20261029T010609.rMWf2s - final - 2026-10-29 04:02:19 +0800
- 主任务: 恢复逐组态 pion 2pt 并绑定 m_pi(config)
- 日志文件: logs/v2026102904.txt
- 文件大小: 4625 bytes
- 触发原因: 任务完成
- 关键内容: 9 组态 pion 2pt、逐组态 m_pi、图件、报告与 264 项验证完成；最终胶子 TMD-PDF 仍待物理闭合
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tmd-c2-v3-20261029 - stage - 2026-10-29 04:51:04 +0800
- 主任务: GPU 生成 9 组态 v3 双极化 C2 并复核 finite-b 审计
- 日志文件: logs/v202610290451.txt
- 文件大小: 2244 bytes
- 触发原因: 阶段成果
- 关键内容: 54/54 v3 C2 完成；source-resolved C2 门通过，full-channel OPE 仍阻塞
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## tmd-finite-b-ope-20261029 - stage - 2026-10-29 06:10:29 +0800
- 主任务: 完成 1D/2D full-channel OPE、论文对照与技能固化
- 日志文件: logs/v202610290610.txt
- 文件大小: 3066 bytes
- 触发原因: 阶段成果
- 关键内容: C2 54/54、1D/2D OPE 9/9、双 audit ready；论文对照和 GPU 技能经验完成
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## resume-20261029T032103.ZSXbIi - final - 2026-10-29 06:28:11 +0800
- 主任务: 恢复 run 20261029T032103.ZSXbIi 并修复 finite-b ratio-v3 的 JK 统计缺陷
- 日志文件: logs/v2026102906.txt
- 文件大小: 5512 bytes
- 触发原因: 任务完成
- 关键内容: 识别 codex run 无可恢复 thread_id；修复 ratio-v3 JK conf 轴被索引吞掉的缺陷并重跑，秩恢复为 8
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tmd-tsep16-audit-20261029 - stage - 2026-10-29 08:17:39 +0800
- 主任务: tsep 扩到 16 的两态联合拟合与全产物数据审计
- 日志文件: logs/v202610290817.txt
- 文件大小: 5207 bytes
- 触发原因: 阶段成果
- 关键内容: C3 网格不变性 0.0；修正 C2 下界后 [5,10] 复现差 0.0；B00 显著性仍 0/12；审计 0 硬错误
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## tmd-tau1-plateau-20261029 - stage - 2026-10-29 08:46:06 +0800
- 主任务: 流窗内 τ=1 的固定插入时间分析与相关组合 P(τ)=B00+B01·exp(−dm·τ)
- 日志文件: logs/v202610290846.txt
- 文件大小: 6216 bytes
- 触发原因: 阶段成果
- 关键内容: τ=1 相关组合 P(1)=0.11σ；信号在 τ≥3 出流窗；审计 NaN 假阳性修复后 0 硬错误；数据集新增 plateau_combination.csv
- 验证: path, size, required sections, conflict markers
- 批次状态: in-progress
## tmd-b7-common-window-20261029 - final - 2026-10-29 10:36:22 +0800
- 主任务: 梯度流胶子准 TMD-PDF：b 扩展到 0..6 与共同流窗流时分析收尾
- 日志文件: logs/v2026102910.txt
- 文件大小: 9762 bytes
- 触发原因: 任务完成
- 关键内容: 27 个 b7 缓存与共同流窗 16 点流时分析验收通过；A/B/C/D 全过、0 硬错误、未重算
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## download-beta-l24x72-source-local-20261029 - final - 2026-10-29 20:02:10 +0800
- 主任务: 检查并修复 L24x72 数据下载脚本在源端环境卡死
- 日志文件: logs/v2026102920.txt
- 文件大小: 6971 bytes
- 触发原因: 任务完成
- 关键内容: 确认默认150项存在、定位SSH认证卡死并新增source-local模式
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## download-l24x72-auth-rollback-20261029 - final - 2026-10-29 20:12:32 +0800
- 主任务: 回滚 L24x72 下载脚本 SSH 二步验证认证模式
- 日志文件: logs/v2026102920-2.txt
- 文件大小: 4326 bytes
- 触发原因: 任务完成
- 关键内容: 恢复 SSH 密码与动态验证码交互认证，保留 source-local 和连接安全参数
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## refer/donghx/_Gradient_Flow_GluonPDF - final - 2026-10-29 20:35:38 +0800
- 主任务: 为 L24x72 下载脚本添加下载进度条与速度显示
- 日志文件: logs/v2026102920-3.txt
- 文件大小: 5776 bytes
- 触发原因: 任务完成
- 关键内容: 新增统一字节进度、实时速度、TTY 自动模式、并行聚合与强制开关；完成语法、串行/并行、TTY、慢速中间进度和远程探针命令验证。
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
## configure-agent-pyqcd-init-20261031 - final - 2026-10-31 23:31:18 +0800
- 主任务: PyQCD diff-init-validate-commit-push-tag-final-log
- 日志文件: logs/v2026103123.txt
- 文件大小: 11714 bytes
- 触发原因: 任务完成
- 关键内容: 初始化 PyQCD 受维护目录治理文件并完成验证、提交、main 推送、dev31 标签和 final 日志
- 验证: path, size, required sections, conflict markers
- 批次状态: complete
