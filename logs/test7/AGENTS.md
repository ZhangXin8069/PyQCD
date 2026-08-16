# AGENTS.md — logs/test7

**test7** —— pyqcd 全功能真实数据实战套件（**服务器正式工作版**，test12 风格）。
输入：正式数据源 `/public/group/lqcd/`（本地与服务器路径一致，本地 10 组态 /
服务器 100+ 组态），正式版 **100 组态**（6250 起、间隔 200 → 6250..26050，
`--start/--step/--n-conf` 可覆盖；默认扫描数据源实际组态）。
实测全部分析功能链：02_ratio → 03_ana_ratio → 04_proton_energy（P2/P0）
→ 06_FH → 05_ana_3dir + 综合报告（LaTeX PDF）。

## 服务器规范（run-local.sh 已内置）

- 环境启动：`source /public/home/zhangxin/mgmt04-env.sh`（存在才 source）。
- GPU：NV-V100-32GB；`env` 自检探测 cupy/CUDA/显存 + nvidia-smi 简报；
  `run --backend auto` 启用 pyqcd cupy 后端（不可用回退 numpy）。
  注意（如实）：分析链统计核心 gvar/lsqfit 为 CPU 库，后端切换不影响其数值路径。
- 组态数据检查机制：makedata 先逐组态检查正式数据源三类原始数据
  （eigensystem/perambulators/configurations 目录与文件数），再检查分析链输入
  corr_pp/ops（形状 (Nt,)/(Nz,Nt)、有限性），异常列明细并中止（exit 1）；
  `env` 预检数据源实际组态齐全度（本地 10 / 服务器 100+）。
- 实时进度日志：python 侧时间戳 + flush（tlog），组态/步骤级进度 + ETA；
  `bash run.sh`（≡ `bash ./run-local.sh --server`）为 nohup 后台模式（日志
  `run-server-<TS>.log`，`tail -f` 实时调控，`kill` 停止）；默认前台 `tee` 落盘。

## 运行

```bash
bash logs/test7/run.sh                 # 正式后台运行入口（≡ bash ./run-local.sh --server）
bash logs/test7/run-local.sh            # 前台一键全链（100 组态正式版）
bash logs/test7/run-local.sh --server   # nohup 后台正式跑
bash logs/test7/run-local.sh --dry-run  # 演练（仅打印命令）
python logs/test7/main.py env                      # 环境/GPU/数据源自检
python logs/test7/main.py makedata                # 检查 + 整理 → input/
python logs/test7/main.py makedata --n-conf 10    # 本地回归（10 组态）
python logs/test7/main.py run --outdir <v<ts>>    # 全功能实战
python logs/test7/main.py run --steps 02_ratio    # 单步骤（调控/断点）
python logs/test7/main.py verify --run-dir <v<ts>># 断言（产物 + 物理合理性）
python logs/test7/main.py check  --run-dir <v<ts>># 断言门（exit 0/1）
python logs/test7/main.py collect --run-dir <v<ts>># 产物清单
```

## 数据源（正式数据源 /public/group/lqcd，本地与服务器路径一致）

- **数据源目录解析**（makedata/env）：`--baseline` > `$test7_BASELINE` >
  `$TEST7_BASELINE` > 默认正式数据源 `/public/group/lqcd`（本地 10 组态 /
  服务器 100+ 组态，仅组态数不同；docker 基线仅可经 `--baseline` 显式指定
  用于本地回归）。
- **三类原始数据**（路径形式与 docker-v20260805/config.py BASE_*_DIR 一致）：
  `eigensystem/{ens}/{conf_id}/`（eigvecs_t{t:03d} × Nt=72）、
  `perambulators/{ens}/light/{conf_id}/`（perams.{cid}.{d}.{t}，d=0..3 × 72）、
  `configurations/CLOVER/{ens}/`（{ens}_cfg_{cid}.lime）。
- **组态自适应**：makedata/env 默认扫描数据源 eigensystem 目录得到实际组态
  （本地 10 / 服务器 100+）；`--start/--step/--n-conf` 显式指定时按序列检查，
  数据缺失如实 FAIL。
- **分析链输入**（corr_pp/ops）：makedata 经 **pyqcd 收缩管线自行计算**
  （从 /public/group/lqcd/ 原始数据出发，vertex→2pt(pp)→OPE，与
  docker-v20260805 数值一致——examples/test0 已验证 237/237、conf6250
  逐位 rel=0；单组态实测 ~20 分钟），计算产物存 `<input>.calc/data/conf<cid>/`
  （docker 布局，供对照），整理后 run 阶段完全脱离数据源目录
  （makedata 已整理 P0/P2 到 data_root，run_04b_p0 从 data_root 读取）。
  耗时：10 组态约 3h（本地），100 组态数小时至 1 天+（服务器 V100）。
  timeout 可经 `test7_MAKEDATA_TIMEOUT` 覆盖（默认 48h）。

## 约定

- **版本目录**：`logs/test7/v<YYYYMMDDHHMM>/`（test12 约定），`--outdir` >
  `$test7_OUTDIR` > `v<ts>/`。
- **数据适配**（makedata）：基线 `corr_pp_P{0,2}` (Nt,) → 平移不变切片矩阵
  (Nt,Nt)；`ops_mu0_nu1/mu3_nu0/mu3_nu1` (Nz,Nt) 原样（组合
  `−O30−O31+2·O01 ≡ ope_combined` 已验证）。P2 2pt 带 phase 负号：
  ratio 负/负相消自洽；能量提取取 |corr2|。
- **组态参数**：`CONF_START=6250, CONF_STEP=200, N_CONF=100`；makedata 默认
  **扫描数据源实际组态**（本地 10 / 服务器 100+），`--start/--step/--n-conf`
  显式指定时按序列检查（数据缺失如实 FAIL）；conf_ids 写入
  `input/data_meta.json`，run/verify/report 全部从 data_meta 读取
  （Nsam 文件名自适应）。
- **物理断言**：P0 meff 平台 ≈ 1.12 GeV（质子质量，docker 基线已验证结论）；
  P2 ≈ 1.56 GeV（色散 E(P2)，与基线 meff_proton_P2 一致）；E0 与 meff 平台一致。
- **局限**：05 三方向与 06 六方向平均在数据可得性限制下退化为单一 z 方向
  （数学流程真实驱动，方向差异物理需三方向数据）。
- 产物：`v<ts>/test7_{summary,timing,verify,collect}.json`、
  `test7_report.pdf`（中文 LaTeX 报告）、106+ 张图（verify F 项断言
  png ≥ 106）、`env.json`。
