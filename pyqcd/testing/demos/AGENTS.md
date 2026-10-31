# AGENTS.md — pyqcd/testing/demos

自包含、可直接运行的 pyqcd 示例。示例只调用生产 API，不复制生产算法。

```bash
python -m pyqcd.testing.demos.tmd_gradient_flow
```

TMD 示例是接口自洽演示，不是物理验收；必须保留 `diagnostic_only` 和非真实闭环边界。
