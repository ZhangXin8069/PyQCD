# AGENTS.md — pyqcd/testing/comparisons

外部参考对照套件，按来源分为 `donghx/`、`lqcddb/`、`supplementary/`，
共享桥接与数据清单位于 `common/`。输出快照统一进入 `artifacts/`，不得改写冻结参考。

```bash
python -m pyqcd.testing.comparisons.main --help
python -m pyqcd.testing.comparisons.main --group all
```

新增用例先核实参考输入存在，再分别报告真实差异、已登记的语义差异和未验证边界。
