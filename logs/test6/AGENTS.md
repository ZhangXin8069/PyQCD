# AGENTS.md — logs/test6

**test6** —— pyqcd 独立复现 `refer/huangcl/04_proton_energy/code_proton_energy.py`
功能测试套件（879 组态三方向 corr2 逐位一致，7 图 + 12 断言）。

被测功能：`pyqcd.analysis._proton_energy`（corr2 → E0 平台拟合 → eff_mass GeV 图，
复用 `_disconnected` 统计基元与 `_plots`/`_fitter`，不 import refer/）。

## 运行

```bash
bash logs/test6/run-local.sh       # 一键：环境自检 → 全链 → 数值比对 → 物理断言
```

## 结构

| 文件 | 内容 |
|---|---|
| `main.py` | 全链（compute → fit → plot，--debug/--parts/--conf-ids 可选） |
| `verify_04_repro.py` | 数值比对（vs `.ref_run/` refer 实跑真值） |
| `run-local.sh` | 一键运行（含 Step 0 环境自检与 Step 3 物理断言） |
| `.ref_run/` | refer 实跑真值（verify 比对基准） |
| `1_result/L24x72/Pz6/` | 产物（corr2_*.npy + 7 图 + 2_fit_report.txt + 1_fit_data.npz） |
| `docs/` | LaTeX 报告（analy_test6_report_20260816.tex/pdf） |

## 约定

- 数据 879 组态三方向（x/y/z/ave），E0(GeV) 用 a⁻¹=1.871 GeV 换算；
  平台拟合 svdcut=1e-6（协方差奇异），物理断言 E0 ≈ 色散预期 3.14 GeV（Pz6）。
