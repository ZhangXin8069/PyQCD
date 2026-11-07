"""pyqcd.einsum 的一键实测接口（PyQCU `testing/simple` 风格）。

每个脚本可独立运行：

    python -m pyqcd.testing.einsum.simple.check_equations
    python -m pyqcd.testing.einsum.simple.check_native --backend cuda
    python -m pyqcd.testing.einsum.simple.check_compat
    python -m pyqcd.testing.einsum.simple.check_memory
    python -m pyqcd.testing.einsum.simple.bench --profile L24 --backend cuda

约定：逐项打印 PASS/FAIL/SKIP，任一 FAIL 时退出码为 1。
"""

__all__ = []
