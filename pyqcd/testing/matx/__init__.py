"""MatX `einsum` 文档契约测试与 cpp-python 桥接口。

本子包独占 `pyqcd/testing/matx/**`，不修改 `pyqcd/einsum/**`、`cpp/cuda/einsum/**`
及生产代码；被测 MatX 语义只按官方文档固化，未核实项一律标 `untested`。

文件分工：

- `_contract.py`：文档语料（`DOC_CASES`）、限制、矛盾登记与 MatX 调用约定接口。
- `_bridge.py`：`cpp/cuda/matx/libpyqcmatx_cuda.so` 的 ctypes 绑定与可用性探针。
- `verify.py`：可执行验证入口（文档用例 vs numpy/pyqcd + 原生 ABI 合约）。

入口：

    python -m pyqcd.testing.matx.verify
    python -m pyqcd.testing.matx.verify --verbose
    python -m pyqcd.testing.matx.verify --json
"""
