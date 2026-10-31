# AGENTS.md — pyqcd/testing/regression/baselines

冻结回归基线根目录。`docker_support/` 保存外部数据接口与运行脚本，
`legacy_pipeline/` 保存自包含冻结蒸馏实现及其已验证输出。

- 已通过的物理结论、源码快照和 `output/` 产物只读。
- 新基线必须使用新目录与来源/提交说明，不能原地覆盖历史证据。
- 验证代码放在 `pyqcd/testing/regression/` 的对应工作流，不修改被比较对象。
