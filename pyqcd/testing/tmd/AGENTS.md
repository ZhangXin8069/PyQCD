# AGENTS.md — pyqcd/testing/tmd

梯度流胶子 TMD-PDF 真实数据工作流、审计脚本与验证入口。

```bash
python -m pyqcd.testing.tmd.pdf_workflow --smoke
python -m pyqcd.testing.tmd.verify_pdf_workflow <run_dir>
```

- 默认输出 `diagnostic_only`；单系综/固定流时/缺少 soft-rapidity 与多尺度数据时不得宣称物理验收。
- 大体积运行产物进入 `artifacts/`，代码不得依赖某个历史运行目录。
- `--dry-run` 仅规划，不得触发 GPU/MPI 或长物理流水线。
