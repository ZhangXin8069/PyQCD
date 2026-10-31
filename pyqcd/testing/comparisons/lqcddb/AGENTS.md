# AGENTS.md — pyqcd/testing/comparisons/lqcddb

PyQCD 对 lqcddb 参考实现的功能映射与数值对照，包括 MPI 搬运层用例。

- 保持参考库只读，差异按“逐位一致 / 约定差异 / 待验证”记录。
- MPI 用例仅在明确提供多个 rank 时运行；普通检查不得隐式启动集群任务。
- 对照结果由父目录 `main.py`/`verify.py` 汇总，不在此目录另造平行入口。
