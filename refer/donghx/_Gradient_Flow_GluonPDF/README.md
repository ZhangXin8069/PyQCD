# 可恢复历史归档

当前主线与数据入口见 [WORKSPACE_INDEX.md](../WORKSPACE_INDEX.md)。

本目录存放停用的历史版本，不是自动可删除的缓存。
归档保留了原代码、日志、结果文件和图片。原路径映射见
[2026-10-27 整理清单](../workspace_cleanup_20261027/archive_manifest.csv)。

历史脚本可能依赖原绝对路径、项目根工作目录或同级文件。
应使用整理工具先检查并恢复原位，再重新运行；恢复不会覆盖现有文件。
纯缓存已删除，可由对应软件从源码／字体／TeX 源文件重新生成。

只预览全部恢复操作：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
python3 -B workspace_cleanup_20261027/cleanup_workspace.py restore
```

恢复指定历史对象：

```bash
python3 -B workspace_cleanup_20261027/cleanup_workspace.py restore \
  --names plot_quark_flow_c3_ratio.py --execute-restore
```

不带 `--names` 并加 `--execute-restore` 会恢复本次全部归档对象。
若任何旧路径已有新文件／目录，工具会拒绝整批操作，避免覆盖。
