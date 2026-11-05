"""pyqcd.einsum 子系统测试与基准组（差分测试 / 实战基准 / 统一入口）。

本子包独占 `pyqcd/testing/einsum/**`，不修改 `pyqcd/einsum/**`、`cpp/**`
及仓库其他文件。被测实现有 bug 时只在报告中登记，不在测试侧绕过或改动它。

入口：
    python -m pyqcd.testing.einsum.main --smoke
    python -m pyqcd.testing.einsum.main --only verify
    python -m pyqcd.testing.einsum.main --only bench
"""
