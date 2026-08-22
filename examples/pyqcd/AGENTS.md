# AGENTS.md — examples/pyqcd

规范化示例与测试（调用 pyqcd 包，不 import refer/）。

```bash
python examples/pyqcd/conftest.py               # 16 项全量测试
python examples/pyqcd/verify_consistency.py     # vs docker-v20260805 基线一致性（A–E 全 0 差异）
python examples/pyqcd/tmd_gradient_flow_demo.py # 梯度流 TMD 全链示例
python examples/pyqcd/dev6/main.py              # dev6：tag-test8 收缩产物 → test0/test6 同类型图表（405 组态）
python examples/pyqcd/dev6/verify_dev6.py examples/pyqcd/dev6/v202608221540   # dev6 断言门（22 项）
python examples/pyqcd/dev7/main.py              # dev7：dev6 收敛迭代（262 组态实际存在扫描 + 02_ratio 链补齐 ratio_3pt 型图）
python examples/pyqcd/dev7/verify_dev7.py examples/pyqcd/dev7/v202608230624   # dev7 断言门（38 项）
```

测试函数定义于 pyqcd/testing/，此处仅导入调度。
