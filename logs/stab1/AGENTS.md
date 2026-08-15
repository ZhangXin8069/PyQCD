# AGENTS.md — logs/stab1

**stab1** —— pyqcd 全功能真实数据实战套件（test12 风格）。
输入：examples/docker-v20260805 基线（10 组态真实格点 QCD 数据）。
实测全部分析功能链：02_ratio → 03_ana_ratio → 04_proton_energy（P2/P0）
→ 06_FH → 05_ana_3dir + 综合报告（LaTeX PDF）。

## 运行

```bash
python logs/stab1/main.py env                      # 环境与数据源自检
python logs/stab1/main.py makedata                 # 整理真实数据 → input/
python logs/stab1/main.py run --outdir <v<ts>>     # 全功能实战 → 版本目录
python logs/stab1/main.py verify --run-dir <v<ts>> # 断言（产物 + 物理合理性）
python logs/stab1/main.py check  --run-dir <v<ts>> # 断言门（exit 0/1）
python logs/stab1/main.py collect --run-dir <v<ts>># 产物清单
bash logs/stab1/run-local.sh                       # 一键全链
```

## 约定

- **版本目录**：`logs/stab1/v<YYYYMMDDHHMM>/`（test12 约定），`--outdir` >
  `$STAB1_OUTDIR` > `v<ts>/`。
- **数据适配**（makedata）：docker 基线 `corr_pp_P{0,2}` (Nt,) → 平移不变
  切片矩阵 (Nt,Nt)；`ops_mu0_nu1/mu3_nu0/mu3_nu1` (Nz,Nt) 原样
  （组合 `−O30−O31+2·O01 ≡ ope_combined` 已验证）。P2 2pt 带 phase 负号：
  ratio 负/负相消自洽；能量提取取 |corr2|。
- **物理断言**：P0 meff 平台 ≈ 1.12 GeV（质子质量，docker 基线已验证结论）；
  P2 ≈ 1.56 GeV（色散 E(P2)，与基线 meff_proton_P2 一致）；E0 与 meff 平台一致。
- **局限**：05 三方向与 06 六方向平均在数据可得性限制下退化为单一 z 方向
  （数学流程真实驱动，方向差异物理需三方向数据）。
- 产物：`v<ts>/stab1_{summary,timing,verify,collect}.json`、
  `stab1_report.pdf`（中文 LaTeX 报告）、106+ 张图、`env.json`。
