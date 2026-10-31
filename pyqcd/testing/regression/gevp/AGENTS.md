# AGENTS.md — pyqcd/testing/regression/gevp

GEVP 文档的数值验证与配图脚本。

```bash
python -m pyqcd.testing.regression.gevp.verify
python -m pyqcd.testing.regression.gevp.make_figures
```

验证须保留解析谱模型、残差/条件数和复数 Hermitian 语义；配图写入 `docs/`，不提交中间缓存。
