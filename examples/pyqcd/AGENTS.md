# AGENTS.md — examples/pyqcd

规范化示例与测试（调用 pyqcd 包，不 import refer/）。

```bash
python examples/pyqcd/conftest.py               # 16 项全量测试
python examples/pyqcd/verify_consistency.py     # vs docker-v20260805 基线一致性（A–E 全 0 差异）
python examples/pyqcd/tmd_gradient_flow_demo.py # 梯度流 TMD 全链示例
```

测试函数定义于 pyqcd/testing/，此处仅导入调度。
