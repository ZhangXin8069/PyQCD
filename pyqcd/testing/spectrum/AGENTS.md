# AGENTS.md — pyqcd/testing/spectrum

有效质量和 ratio 工作流，复用 `pyqcd.analysis` 完成扫描、重采样、拟合、成图与独立验证。

```bash
python -m pyqcd.testing.spectrum.effective_mass
python -m pyqcd.testing.spectrum.verify_effective_mass <run_dir>
python -m pyqcd.testing.spectrum.effective_mass_ratio
python -m pyqcd.testing.spectrum.verify_effective_mass_ratio <run_dir>
```

运行产物进入 `artifacts/`。输入不足的通道必须明确跳过并记录原因，不得补造图或拟合值。
