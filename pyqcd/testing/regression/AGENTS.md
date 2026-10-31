# AGENTS.md — pyqcd/testing/regression

回归测试与冻结基线。通用一致性入口：

```bash
python -m pyqcd.testing.regression.consistency
```

- `baselines/` 是只读参考，不能为让测试通过而改期望值。
- 各组回归的子目录必须提供自己的入口/运行说明。
- 依赖缺失的生成数据应明确跳过或报缺，不能以空数组伪造一致性。
