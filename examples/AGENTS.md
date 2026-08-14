# AGENTS.md — examples

| 目录 | 内容 |
|---|---|
| `docker-v20260805/` | ★ 成功实例基线（10 组态全量蒸馏 GPU 管线，勿改已验证物理结论） |
| `_docker/` | 启动器与数据下载/打包脚本、历史报告 |
| `pyqcd/` | 规范化示例与测试（调用 pyqcd 包） |

## pyqcd 示例

```bash
python examples/pyqcd/conftest.py                 # 16 项测试（γ基/重整化/梯度流/TMD/匹配/混合/提取链等）
python examples/pyqcd/tmd_gradient_flow_demo.py   # 梯度流重整化胶子 TMD 全链示例
```

## 成功实例（docker-v20260805）关键物理结论（勿重复调试）

- **pn 2pt = 0**：质子(uud)↔中子(udd) 味不守恒，Wick 无有效图。物理正确。
- **质子质量 ~1.12 GeV**（非 1.0）：该系综夸克重（m_π≈0.286）。
- **OPE 已验证**：与 v20260802 相关系数 1.0；`.lime` 文件 136 字节 trailer。
- 不相连比值在 10 组态下噪声大；连通 3pt/2pt 比值（pion P0 ≈ −0.96）为干净结果。
