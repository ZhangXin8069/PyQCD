# PyQCD form 治理审计（2026-10-27）

## 范围

本次审计服务于胶子 TMD-PDF 失败关闭诊断交付的 Git 发布，只检查本轮改动、
日志、数据运行归档及仓库根 `AGENTS.md` 的格式契约，不扩大修改历史文档或只读参考树。

## 验证

- `bash -n skills/form/scripts/form-audit.sh`：PASS。
- `git diff --check`：PASS。
- `skills/form/scripts/form-audit.sh`：FAIL，`unexpected findings=3`。

## 既有问题

以下三项均来自提交 `8eea2e8d`，当前工作区无修改：

- `docs/absorb_dev20_gluon_tmd_audit_20261022.md`：路径包含 Git 标签名。
- `docs/plans/2026-10-22-absorb-dev20-gluon-tmd-debug.md`：路径包含 Git 标签名。
- `refer/git-rep/README.md`：未记录子模块 PyQCU 的来源
  `https://gitee.com/zhangxin8069/PyQCU.git`。

前两项属于历史文档命名债，后一项位于只读 `refer/` 边界。为避免超越本次发布范围，
本次不改名、不移动、不修改参考树，也不修改审计包装器来隐藏问题。

## 恢复方法

后续应由文档治理任务重命名两份历史文档并同步全部引用；由获得 `refer/` 修改授权的任务
补齐 PyQCU 来源。完成后再运行本地 `form` 审计，目标为 `0 unexpected findings`。

## 本轮交付判定

本轮新增数据运行归档符合 `data/**` 具名例外；日志符合 `logs/` 文本格式；生产代码与
测试未引入新的命名或目录违规。由于上述既有问题，`form` 全局门禁保持红色并如实登记。
